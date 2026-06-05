# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""VACE variant of Wan21Transformer3DModel for conditional video generation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import (
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard
from vllm_omni.diffusion.forward_context import get_forward_context
from .wan2_1_transformer import (
    Transformer2DModelOutput,
    Wan21Transformer3DModel,
    WanTransformerBlock,
)


class VaceWan21TransformerBlock(WanTransformerBlock):
    """VACE variant of WanTransformerBlock with proj_in/proj_out for skip connections."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
        block_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            dim,
            ffn_dim,
            num_heads,
            eps,
            added_kv_proj_dim,
            cross_attn_norm,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.proj_in = nn.Linear(dim, dim) if block_id == 0 else None
        self.proj_out = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        hidden_states_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proj_in is not None:
            control_hidden_states = self.proj_in(control_hidden_states)
            control_hidden_states = control_hidden_states + hidden_states

        control_hidden_states = super().forward(
            control_hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb,
            hidden_states_mask,
        )

        conditioning_states = self.proj_out(control_hidden_states)
        return conditioning_states, control_hidden_states


class Wan21VACETransformer3DModel(Wan21Transformer3DModel):
    """VACE-extended Wan2.1 transformer with conditioning blocks."""

    _layerwise_offload_blocks_attrs = ["blocks", "vace_blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return (
            name.startswith("blocks.") or name.startswith("vace_blocks.")
        ) and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    # Shard hidden_states before VACE blocks (replaces parent's blocks.0)
    _sp_plan = {
        **{k: v for k, v in Wan21Transformer3DModel._sp_plan.items() if k != "blocks.0"},
        "_sp_shard_point": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
        },
    }

    def __init__(
        self,
        *,
        vace_layers: list[int] | None = None,
        vace_in_channels: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__(quant_config=quant_config, prefix=prefix, **kwargs)

        if vace_layers is None:
            vace_layers = [0, 5, 10, 15, 20, 25, 30, 35]
        if max(vace_layers) >= self.config.num_layers:
            raise ValueError(
                f"VACE layers {vace_layers} exceed the number of transformer layers "
                f"{self.config.num_layers}."
            )
        if 0 not in vace_layers:
            raise ValueError("VACE layers must include layer 0.")

        self.vace_layers = list(vace_layers)
        self.vace_layers_mapping = {
            layer_idx: vace_idx for vace_idx, layer_idx in enumerate(vace_layers)
        }
        vace_in_channels = vace_in_channels or 96
        self.config.vace_layers = self.vace_layers
        self.config.vace_in_channels = vace_in_channels

        # SP shard point: Identity module that _sp_plan hooks into to shard
        # hidden_states before VACE processing (instead of at blocks.0)
        self._sp_shard_point = nn.Identity()

        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.vace_patch_embedding = nn.Conv3d(
            vace_in_channels,
            inner_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.vace_blocks = nn.ModuleList(
            [
                VaceWan21TransformerBlock(
                    inner_dim,
                    self.config.ffn_dim,
                    self.config.num_attention_heads,
                    self.config.eps,
                    self.config.added_kv_proj_dim,
                    self.config.cross_attn_norm,
                    block_id=i,
                    quant_config=quant_config,
                    prefix=f"{prefix}.vace_blocks.{i}" if prefix else f"vace_blocks.{i}",
                )
                for i in range(len(vace_layers))
            ]
        )

        # ROPE helper
        self._cached_rope_emb = None
        self._cached_rope_resolution = None

    def embed_vace_context(
        self,
        vace_context: torch.Tensor,
        seq_len: int,
        sp_size: int = 1,
    ) -> torch.Tensor:
        """Compute VACE patch embeddings, aligned and sharded for SP.

        Args:
            vace_context: Raw conditioning tensor [B, C, T, H, W].
            seq_len: Target full (padded) sequence length to align to.
            sp_size: Sequence parallel world size.
        """
        vace_embeds = self.vace_patch_embedding(vace_context)
        vace_embeds = vace_embeds.flatten(2).transpose(1, 2)

        # Align to target seq_len (may include SP padding)
        if vace_embeds.size(1) < seq_len:
            vace_embeds = F.pad(vace_embeds, (0, 0, 0, seq_len - vace_embeds.size(1)))

        if sp_size > 1:
            vace_embeds = sp_shard(vace_embeds, dim=1)
        return vace_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        vace_context: torch.Tensor | None = None,
        vace_context_scale: float | list[float] = 1.0,
        control_hidden_states: torch.Tensor | None = None,
        control_hidden_states_scale: torch.Tensor | float | list[float] | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput | IntermediateTensors:
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Compute RoPE embeddings (sharded by _sp_plan via split_output=True)
        current_rope_resolution = (post_patch_num_frames, post_patch_height, post_patch_width)
        if self._cached_rope_resolution == current_rope_resolution and self._cached_rope_emb is not None:
            rotary_emb = self._cached_rope_emb
        else:
            freqs_cos, freqs_sin = self.rope(hidden_states)
            rotary_emb = (freqs_cos[..., 0::2].to(hidden_states.dtype), freqs_sin[..., 1::2].to(hidden_states.dtype))
            self._hidden_states_shape = hidden_states.shape
            self._cached_rope_emb = rotary_emb
            self._cached_rope_resolution = current_rope_resolution

        if is_pipeline_first_stage():
            # Patch embedding and flatten to sequence.
            hidden_states = self.patch_embedding(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        else:
            if intermediate_tensors is None:
                raise RuntimeError("intermediate_tensors must be provided for non-first PP stages")
            hidden_states = intermediate_tensors["hidden_states"]

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        timestep_proj = self.timestep_proj_prepare(timestep_proj, ts_seq_len)

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if vace_context is None:
            vace_context = control_hidden_states
        if control_hidden_states_scale is not None:
            vace_context_scale = control_hidden_states_scale

        if is_pipeline_first_stage():
            # Shard hidden_states via _sp_plan hook (before VACE, not at blocks.0).
            hidden_states = self._sp_shard_point(hidden_states)

        # SP state and attention mask for padding
        hidden_states_mask = None
        ctx = get_forward_context()
        parallel_config = ctx.omni_diffusion_config.parallel_config
        sp_size = parallel_config.sequence_parallel_size if parallel_config is not None else 1
        if ctx.sp_original_seq_len is not None and ctx.sp_padding_size > 0:
            padded_seq_len = ctx.sp_original_seq_len + ctx.sp_padding_size
            hidden_states_mask = torch.ones(
                batch_size,
                padded_seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            hidden_states_mask[:, ctx.sp_original_seq_len :] = False

        # VACE: embed context and run conditioning blocks
        vace_hints = None
        if vace_context is not None and is_pipeline_first_stage():
            full_seq_len = hidden_states.shape[1] * sp_size
            control_hidden_states = self.embed_vace_context(vace_context.to(hidden_states.dtype), full_seq_len, sp_size)
            vace_hints = []
            for i, block in enumerate(self.vace_blocks):
                conditioning_states, control_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    control_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    hidden_states_mask,
                )
                vace_hints.append(conditioning_states)
        elif vace_context is not None and intermediate_tensors is not None:
            try:
                vace_hints = [
                    intermediate_tensors[f"vace_hint_{i}"]
                    for i in range(len(self.vace_layers))
                ]
            except KeyError as exc:
                raise RuntimeError(
                    "vace_hints must be provided for non-first VACE PP stages"
                ) from exc

        # Normalize scale to per-layer list
        if vace_hints is not None and isinstance(vace_context_scale, (int, float)):
            vace_context_scale = [vace_context_scale] * len(vace_hints)

        # Transformer blocks with VACE hint application
        for i, block in enumerate(
            self.blocks[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                hidden_states_mask,
            )
            if vace_hints is not None and self.vace_layers_mapping is not None and i in self.vace_layers_mapping:
                vace_idx = self.vace_layers_mapping[i]
                hidden_states = hidden_states + vace_hints[vace_idx] * vace_context_scale[vace_idx]

        if not is_pipeline_last_stage():
            tensors = {"hidden_states": hidden_states}
            if vace_hints is not None:
                tensors.update({f"vace_hint_{i}": hint for i, hint in enumerate(vace_hints)})
            return IntermediateTensors(tensors)

        # Output norm, projection & unpatchify
        shift, scale = self.output_scale_shift_prepare(temb)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        if shift.ndim == 2:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)

        hidden_states = self.norm_out(hidden_states, scale, shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
