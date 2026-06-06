# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any, ClassVar, cast

import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    UMT5EncoderModel,
)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import (
    AsyncLatents,
    PipelineParallelMixin,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.interface import (
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from .wan2_1_transformer import Wan21Transformer3DModel
from .wan2_1_vace_transformer import Wan21VACETransformer3DModel
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

WAN21_SAMPLE_SOLVER_CHOICES = {"unipc"}


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def load_transformer_config(
    model_path: str,
    subfolder: str = "transformer",
    local_files_only: bool = True,
) -> dict[str, Any]:
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(
        repo_id=model_path,
        filename=f"{subfolder}/config.json",
    )
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _make_wan_transformer_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "attention_head_dim",
        "added_kv_proj_dim",
        "cross_attn_norm",
        "eps",
        "ffn_dim",
        "freq_dim",
        "image_dim",
        "in_channels",
        "num_attention_heads",
        "num_layers",
        "out_channels",
        "pos_embed_seq_len",
        "rope_max_seq_len",
        "text_dim",
    }
    kwargs = {key: config[key] for key in keys if key in config}
    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    return kwargs


def create_transformer_from_config(
    config: dict[str, Any],
    quant_config=None,
    prefix: str = "",
) -> Wan21Transformer3DModel:
    if "quantization_config" in config:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(
            quant_config,
            config["quantization_config"],
        )
    return Wan21Transformer3DModel(
        **_make_wan_transformer_kwargs(config),
        quant_config=quant_config,
        prefix=prefix,
    )


def create_vace_transformer_from_config(
    config: dict[str, Any],
    quant_config=None,
    prefix: str = "",
) -> Wan21VACETransformer3DModel:
    kwargs = _make_wan_transformer_kwargs(config)
    for key in ("vace_in_channels", "vace_layers"):
        if key in config:
            kwargs[key] = config[key]
    if "quantization_config" in config:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(
            quant_config,
            config["quantization_config"],
        )
    return Wan21VACETransformer3DModel(
        **kwargs,
        quant_config=quant_config,
        prefix=prefix,
    )


def _ensure_wan21_transformer_metadata(transformer: nn.Module) -> None:
    block_attrs = getattr(transformer.__class__, "_layerwise_offload_blocks_attrs", None)
    if block_attrs:
        return
    if hasattr(transformer, "blocks"):
        transformer.__class__._layerwise_offload_blocks_attrs = ["blocks"]


def build_wan21_scheduler(flow_shift: float) -> FlowUniPCMultistepScheduler:
    return FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=flow_shift,
        prediction_type="flow_prediction",
    )


def resolve_wan21_default_flow_shift(model: str | None) -> float:
    if model:
        normalized = str(model).replace("\\", "/").lower()
        if "flf2v" in normalized:
            return 16.0
        if "i2v" in normalized and "720p" in normalized:
            return 5.0
    return 3.0


def resolve_wan21_sample_solver(
    req: OmniDiffusionRequest,
    default: str = "unipc",
) -> str:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw = extra_args.get("sample_solver", default)
    sample_solver = str(raw).strip().lower()
    if sample_solver not in WAN21_SAMPLE_SOLVER_CHOICES:
        raise ValueError(
            f"Invalid Wan2.1 sample_solver={raw!r}. "
            f"Expected one of: {sorted(WAN21_SAMPLE_SOLVER_CHOICES)}"
        )
    return sample_solver


def resolve_wan21_flow_shift(
    req: OmniDiffusionRequest,
    od_config: OmniDiffusionConfig,
) -> float:
    extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
    raw_flow_shift = extra_args.get("flow_shift")
    if raw_flow_shift is None:
        raw_flow_shift = (
            od_config.flow_shift
            if od_config.flow_shift is not None
            else resolve_wan21_default_flow_shift(getattr(od_config, "model", None))
        )
    try:
        return float(raw_flow_shift)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid flow_shift={raw_flow_shift!r}; expected float.") from exc


def _wan21_has_unsupported_guidance_scale_2(req: OmniDiffusionRequest) -> bool:
    guidance_scale_2 = req.sampling_params.guidance_scale_2
    if guidance_scale_2 is None:
        return False

    try:
        return abs(
            float(guidance_scale_2) - float(req.sampling_params.guidance_scale)
        ) > 1e-6
    except (TypeError, ValueError):
        return True


def _prompt_text(prompt: OmniTextPrompt | str) -> str | None:
    return prompt if isinstance(prompt, str) else prompt.get("prompt")


def _negative_prompt_text(prompt: OmniTextPrompt | str) -> str | None:
    return None if isinstance(prompt, str) else prompt.get("negative_prompt")


def _multi_modal_data(prompt: OmniTextPrompt | str) -> dict[str, Any] | None:
    return None if isinstance(prompt, str) else prompt.get("multi_modal_data", {})


def _ensure_prompt_dict(prompt: OmniTextPrompt | str) -> OmniTextPrompt:
    if isinstance(prompt, str):
        return OmniTextPrompt(prompt=prompt)
    return prompt


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _load_pil_image(raw_image: str | PIL.Image.Image) -> PIL.Image.Image:
    if isinstance(raw_image, str):
        return PIL.Image.open(raw_image).convert("RGB")
    return raw_image.convert("RGB")


def _default_i2v_area(model_name: str) -> int:
    return 720 * 1280 if "720" in model_name.lower() else 480 * 832


def _resize_to_area(
    image: PIL.Image.Image,
    max_area: int,
    mod_value: int = 16,
) -> tuple[int, int]:
    import numpy as np

    aspect_ratio = image.height / image.width
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return max(height, mod_value), max(width, mod_value)


def get_wan21_video_post_process_func(od_config: OmniDiffusionConfig):
    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
        sampling_params=None,
    ):
        if output_type == "latent":
            return video

        custom_output: dict[str, Any] = {}
        if sampling_params is not None and getattr(
            sampling_params,
            "enable_frame_interpolation",
            False,
        ):
            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            custom_output["video_fps_multiplier"] = multiplier

        if isinstance(video, torch.Tensor):
            video = video_processor.postprocess_video(video, output_type=output_type)

        return {"video": video, "custom_output": custom_output}

    return post_process_func


def get_wan21_post_process_func(od_config: OmniDiffusionConfig):
    return get_wan21_video_post_process_func(od_config)


def get_wan21_i2v_post_process_func(od_config: OmniDiffusionConfig):
    return get_wan21_video_post_process_func(od_config)


def get_wan21_flf2v_post_process_func(od_config: OmniDiffusionConfig):
    return get_wan21_video_post_process_func(od_config)


def get_wan21_vace_post_process_func(od_config: OmniDiffusionConfig):
    return get_wan21_video_post_process_func(od_config)


def get_wan21_pre_process_func(od_config: OmniDiffusionConfig):
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            request.prompts[i] = _ensure_prompt_dict(prompt)
        return request

    return pre_process_func


def get_wan21_i2v_pre_process_func(od_config: OmniDiffusionConfig):
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            prompt = _ensure_prompt_dict(prompt)
            multi_modal_data = prompt.setdefault("multi_modal_data", {})
            raw_image = multi_modal_data.get("image")
            if raw_image is None:
                request.prompts[i] = prompt
                continue
            if isinstance(raw_image, list):
                if len(raw_image) != 1:
                    logger.warning(
                        "Wan2.1 I2V accepts a single image. Using the first image."
                    )
                raw_image = raw_image[0]
            if not isinstance(raw_image, (str, PIL.Image.Image)):
                raise TypeError(
                    f"Unsupported image format {raw_image.__class__}. "
                    'Please use `"multi_modal_data": {"image": ...}` with '
                    "a path or PIL.Image.Image."
                )
            image = _load_pil_image(raw_image)
            if request.sampling_params.height is None or request.sampling_params.width is None:
                height, width = _resize_to_area(
                    image,
                    _default_i2v_area(od_config.model),
                )
                if request.sampling_params.height is None:
                    request.sampling_params.height = height
                if request.sampling_params.width is None:
                    request.sampling_params.width = width
            image = image.resize(
                (
                    cast(int, request.sampling_params.width),
                    cast(int, request.sampling_params.height),
                ),
                PIL.Image.Resampling.LANCZOS,
            )
            multi_modal_data["image"] = image
            request.prompts[i] = prompt
        return request

    return pre_process_func


def _normalize_wan21_flf2v_images(
    multi_modal_data: dict[str, Any],
) -> tuple[PIL.Image.Image, PIL.Image.Image]:
    raw_image = multi_modal_data.get("image")
    raw_last_image = multi_modal_data.get("last_image")

    if isinstance(raw_image, list):
        if raw_last_image is not None:
            raise ValueError(
                "Wan2.1 FLF2V accepts either image as [first, last] or last_image, "
                "not both."
            )
        if len(raw_image) != 2:
            raise ValueError(
                "Wan2.1 FLF2V image list must contain exactly two images: "
                "[first, last]."
            )
        first_raw_image = raw_image[0]
        last_raw_image = raw_image[1]
    else:
        first_raw_image = raw_image
        last_raw_image = raw_last_image

    if first_raw_image is None or last_raw_image is None:
        raise ValueError("Wan2.1 FLF2V requires both first and last images.")

    for raw in (first_raw_image, last_raw_image):
        if not isinstance(raw, (str, PIL.Image.Image)):
            raise TypeError(
                f"Unsupported image format {raw.__class__}. "
                'Please use `"multi_modal_data": {"image": ..., "last_image": ...}` '
                'or `"multi_modal_data": {"image": [first, last]}` with paths or '
                "PIL.Image.Image values."
            )

    first_image = _load_pil_image(first_raw_image)
    last_image = _load_pil_image(last_raw_image)
    multi_modal_data["image"] = first_image
    multi_modal_data["last_image"] = last_image
    return first_image, last_image


def _prepare_wan21_flf2v_image_embeds(
    image_embeds: torch.Tensor,
    prompt_batch_size: int,
) -> torch.Tensor:
    if image_embeds.ndim != 3:
        raise ValueError(
            "Wan2.1 FLF2V image_embeds must be a 3D tensor with shape [2, S, D] "
            "or [2B, S, D]."
        )

    image_batch_size = image_embeds.shape[0]
    expanded_image_batch_size = prompt_batch_size * 2
    if image_batch_size == 2:
        return image_embeds.repeat(prompt_batch_size, 1, 1)
    if image_batch_size == expanded_image_batch_size:
        return image_embeds

    raise ValueError(
        "Wan2.1 FLF2V image_embeds must have shape [2, S, D] for first/last "
        f"images or [2B, S, D] for an expanded prompt batch; got first dimension "
        f"{image_batch_size} for prompt batch {prompt_batch_size}. Pre-merged "
        "[B, 2S, D] embeddings are not supported."
    )


def get_wan21_flf2v_pre_process_func(od_config: OmniDiffusionConfig):
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            prompt = _ensure_prompt_dict(prompt)
            multi_modal_data = prompt.setdefault("multi_modal_data", {})
            first_image, last_image = _normalize_wan21_flf2v_images(multi_modal_data)
            if request.sampling_params.height is None or request.sampling_params.width is None:
                height, width = _resize_to_area(
                    first_image,
                    _default_i2v_area(od_config.model),
                )
                if request.sampling_params.height is None:
                    request.sampling_params.height = height
                if request.sampling_params.width is None:
                    request.sampling_params.width = width
            size = (
                cast(int, request.sampling_params.width),
                cast(int, request.sampling_params.height),
            )
            multi_modal_data["image"] = first_image.resize(
                size,
                PIL.Image.Resampling.LANCZOS,
            )
            multi_modal_data["last_image"] = last_image.resize(
                size,
                PIL.Image.Resampling.LANCZOS,
            )
            request.prompts[i] = prompt
        return request

    return pre_process_func


def get_wan21_vace_pre_process_func(od_config: OmniDiffusionConfig):
    def load_image_like(value):
        if value is None:
            return None
        if isinstance(value, str):
            return PIL.Image.open(value).convert("RGB")
        if isinstance(value, PIL.Image.Image):
            return value.convert("RGB")
        return value

    def load_sequence(value):
        if value is None:
            return None
        if isinstance(value, (str, PIL.Image.Image)):
            return [load_image_like(value)]
        if isinstance(value, list):
            return [load_image_like(item) for item in value]
        return value

    def load_mask_like(value):
        if value is None:
            return None
        if isinstance(value, str):
            return PIL.Image.open(value).convert("L")
        if isinstance(value, PIL.Image.Image):
            return value.convert("L")
        return value

    def load_mask_sequence(value):
        if value is None:
            return None
        if isinstance(value, (str, PIL.Image.Image)):
            return [load_mask_like(value)]
        if isinstance(value, list):
            return [load_mask_like(item) for item in value]
        return value

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            prompt = _ensure_prompt_dict(prompt)
            multi_modal_data = prompt.get("multi_modal_data", {}) or {}
            additional_information = prompt.setdefault("additional_information", {})
            additional_information["source_video"] = load_sequence(
                _first_not_none(
                    multi_modal_data.get("video"),
                    additional_information.get("source_video"),
                    additional_information.get("video"),
                )
            )
            additional_information["mask"] = load_mask_sequence(
                _first_not_none(
                    multi_modal_data.get("mask"),
                    additional_information.get("mask"),
                )
            )
            additional_information["reference_images"] = load_sequence(
                _first_not_none(
                    multi_modal_data.get("reference_images"),
                    multi_modal_data.get("image"),
                    additional_information.get("reference_images"),
                )
            )
            request.prompts[i] = prompt
        return request

    return pre_process_func


class Wan21PipelineBase(
    nn.Module,
    SupportsComponentDiscovery,
    PipelineParallelMixin,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    support_image_input = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.prefix = prefix
        self.model = od_config.model
        self.local_files_only = os.path.exists(self.model)
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        if od_config.boundary_ratio is not None:
            raise ValueError("Wan2.1 has a single transformer and does not support boundary_ratio.")

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        prefetch_subfolders(
            self.model,
            ["tokenizer", "text_encoder", "vae"],
            local_files_only=self.local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            subfolder="tokenizer",
            local_files_only=self.local_files_only,
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.model,
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            self.model,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
        ).to(self.device)
        transformer_config = load_transformer_config(
            self.model,
            "transformer",
            self.local_files_only,
        )
        with set_current_diffusion_config(od_config):
            self.transformer = self._create_transformer(transformer_config)
        _ensure_wan21_transformer_metadata(self.transformer)
        self.transformer_config = self.transformer.config

        self._sample_solver = "unipc"
        self._flow_shift = (
            od_config.flow_shift
            if od_config.flow_shift is not None
            else resolve_wan21_default_flow_shift(self.model)
        )
        self.scheduler = build_wan21_scheduler(self._flow_shift)
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    def _create_transformer(self, config: dict[str, Any]) -> Wan21Transformer3DModel:
        quant_config = getattr(self.od_config, "quantization_config", None)
        return create_transformer_from_config(
            config,
            quant_config=quant_config,
            prefix=f"{self.prefix}transformer",
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def _cache_context(self, name: str):
        cache_context = getattr(self.transformer, "cache_context", None)
        return cache_context(name) if callable(cache_context) else nullcontext()

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)
        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(
                ids_neg.to(device),
                mask_neg.to(device),
            ).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [
                u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)
            ]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1,
                num_videos_per_prompt,
                1,
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_videos_per_prompt,
                seq_len,
                -1,
            )
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Generator list length {len(generator)} does not match batch size {batch_size}."
            )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def predict_noise(
        self,
        current_model: nn.Module | None = None,
        cache_name: str = "cond",
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        current_model = current_model or self.transformer
        cache_context = getattr(current_model, "cache_context", None)
        context = cache_context(cache_name) if callable(cache_context) else nullcontext()
        with context:
            result = current_model(**kwargs)
        if isinstance(result, IntermediateTensors):
            return result
        return result[0] if isinstance(result, tuple) else result.sample

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any] | None,
        extra_model_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | AsyncLatents:
        attention_kwargs = attention_kwargs or {}
        extra_model_kwargs = extra_model_kwargs or {}
        try:
            with self.progress_bar(total=len(timesteps)) as pbar:
                for step_idx, t in enumerate(timesteps):
                    self._current_timestep = t
                    set_forward_context_denoise_step_idx(step_idx)
                    latent_model_input = latents.to(dtype)
                    timestep = t.expand(latents.shape[0])
                    do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
                    positive_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "current_model": self.transformer,
                        "cache_name": "cond",
                        **extra_model_kwargs,
                    }
                    negative_kwargs = None
                    if do_true_cfg:
                        negative_kwargs = {
                            "hidden_states": latent_model_input,
                            "timestep": timestep,
                            "encoder_hidden_states": negative_prompt_embeds,
                            "attention_kwargs": attention_kwargs,
                            "return_dict": False,
                            "current_model": self.transformer,
                            "cache_name": "uncond",
                            **extra_model_kwargs,
                        }
                    noise_pred = self.predict_noise_maybe_with_cfg(
                        do_true_cfg=do_true_cfg,
                        true_cfg_scale=guidance_scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=False,
                    )
                    latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                    pbar.update()
            return latents
        finally:
            set_forward_context_denoise_step_idx(None)

    def _decode_latents(
        self,
        latents: torch.Tensor,
        output_type: str | None,
    ) -> torch.Tensor:
        if output_type == "latent":
            return latents
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1,
            self.vae.config.z_dim,
            1,
            1,
            1,
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        return self.vae.decode(latents, return_dict=False)[0]

    def _prepare_common_forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None,
        negative_prompt: str | None,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        frame_num: int,
        generator: torch.Generator | list[torch.Generator] | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
    ):
        if len(req.prompts) != 1:
            raise ValueError("Wan2.1 currently supports a single prompt per request.")
        first_prompt = req.prompts[0]
        prompt = _prompt_text(first_prompt) if prompt is None else prompt
        negative_prompt = (
            _negative_prompt_text(first_prompt) if negative_prompt is None else negative_prompt
        )
        if not isinstance(first_prompt, str):
            prompt_embeds = (
                prompt_embeds
                if prompt_embeds is not None
                else first_prompt.get("prompt_embeds")
            )
            negative_prompt_embeds = (
                negative_prompt_embeds
                if negative_prompt_embeds is not None
                else first_prompt.get("negative_prompt_embeds")
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for Wan2.1 generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames or frame_num
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]
        height = max(mod_value, (height // mod_value) * mod_value)
        width = max(mod_value, (width // mod_value) * mod_value)
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.boundary_ratio is not None:
            raise ValueError("Wan2.1 does not support boundary_ratio.")
        if _wan21_has_unsupported_guidance_scale_2(req):
            raise ValueError("Wan2.1 does not support guidance_scale_2.")
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        self._guidance_scale = cast(float, guidance_scale)

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        sample_solver = resolve_wan21_sample_solver(req, default=self._sample_solver)
        flow_shift = resolve_wan21_flow_shift(req, self.od_config)
        if sample_solver != self._sample_solver or abs(flow_shift - self._flow_shift) > 1e-6:
            self.scheduler = build_wan21_scheduler(flow_shift)
            self._sample_solver = sample_solver
            self._flow_shift = flow_shift

        dtype = self.transformer.dtype
        do_cfg = self._guidance_scale > 1.0
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=cast(str, prompt),
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
                max_sequence_length=req.sampling_params.max_sequence_length or 512,
                device=self.device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=self.device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    device=self.device,
                    dtype=dtype,
                )
            elif do_cfg:
                raise ValueError(
                    "negative_prompt_embeds must be provided when prompt_embeds are "
                    "given and guidance_scale > 1."
                )

        self.scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            height,
            width,
            num_frames,
            timesteps,
            generator,
            dtype,
        )

    @torch.inference_mode()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        (
            prompt_embeds,
            negative_prompt_embeds,
            height,
            width,
            num_frames,
            timesteps,
            generator,
            dtype,
        ) = self._prepare_common_forward(
            req,
            prompt,
            negative_prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            frame_num,
            generator,
            prompt_embeds,
            negative_prompt_embeds,
        )
        latents = self.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=self.transformer_config.in_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=req.sampling_params.latents,
        )
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=cast(float, self._guidance_scale),
            dtype=dtype,
            attention_kwargs=attention_kwargs,
        )
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        output = self._decode_latents(latents, output_type)
        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class Wan21I2VPipelineBase(Wan21PipelineBase, SupportImageInput):
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "image_encoder"]
    support_image_input = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        prefetch_subfolders(
            self.model,
            ["image_encoder", "image_processor"],
            local_files_only=self.local_files_only,
        )
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.model,
            subfolder="image_processor",
            local_files_only=self.local_files_only,
        )
        self.image_encoder = CLIPVisionModel.from_pretrained(
            self.model,
            subfolder="image_encoder",
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
        ).to(self.device)

    def encode_image(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image] | torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        device = device or self.device
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def prepare_i2v_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (
            batch_size,
            self.vae.config.z_dim,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)],
            dim=2,
        )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)
        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)
        latent_condition = latent_condition.to(dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latent_condition.device, latent_condition.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1,
            self.vae.config.z_dim,
            1,
            1,
            1,
        ).to(latent_condition.device, latent_condition.dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        mask_lat_size[:, :, list(range(1, num_frames))] = 0
        first_frame_mask = torch.repeat_interleave(
            mask_lat_size[:, :, 0:1],
            dim=2,
            repeats=self.vae_scale_factor_temporal,
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size,
            -1,
            self.vae_scale_factor_temporal,
            latent_height,
            latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(1, 2).to(latent_condition.device)
        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    @torch.inference_mode()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        (
            prompt_embeds,
            negative_prompt_embeds,
            height,
            width,
            num_frames,
            timesteps,
            generator,
            dtype,
        ) = self._prepare_common_forward(
            req,
            prompt,
            negative_prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            frame_num,
            generator,
            prompt_embeds,
            negative_prompt_embeds,
        )
        multi_modal_data = _multi_modal_data(req.prompts[0]) or {}
        raw_image = multi_modal_data.get("image")
        if raw_image is None:
            raise ValueError(
                "Wan2.1 I2V requires an image; image_embeds alone are not sufficient."
            )
        if isinstance(raw_image, list):
            raw_image = raw_image[0]
        image = cast(PIL.Image.Image | torch.Tensor, raw_image)
        if image_embeds is None:
            image_embeds = self.encode_image(image, self.device)
        image_embeds = image_embeds.repeat(prompt_embeds.shape[0], 1, 1).to(dtype)

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
            image_tensor = video_processor.preprocess(image, height=height, width=width)
        else:
            image_tensor = image
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        latents, condition = self.prepare_i2v_latents(
            image=image_tensor,
            batch_size=prompt_embeds.shape[0],
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=req.sampling_params.latents,
        )
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=cast(float, self._guidance_scale),
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            extra_model_kwargs={
                "encoder_hidden_states_image": image_embeds,
                "condition": condition,
            },
        )
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        output = self._decode_latents(latents, output_type)
        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any] | None,
        extra_model_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | AsyncLatents:
        attention_kwargs = attention_kwargs or {}
        extra_model_kwargs = extra_model_kwargs or {}
        condition = extra_model_kwargs.pop("condition")
        try:
            with self.progress_bar(total=len(timesteps)) as pbar:
                for step_idx, t in enumerate(timesteps):
                    self._current_timestep = t
                    set_forward_context_denoise_step_idx(step_idx)
                    latent_model_input = torch.cat([latents, condition], dim=1).to(dtype)
                    timestep = t.expand(latents.shape[0])
                    do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
                    positive_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "current_model": self.transformer,
                        "cache_name": "cond",
                        **extra_model_kwargs,
                    }
                    negative_kwargs = None
                    if do_true_cfg:
                        negative_kwargs = {
                            "hidden_states": latent_model_input,
                            "timestep": timestep,
                            "encoder_hidden_states": negative_prompt_embeds,
                            "attention_kwargs": attention_kwargs,
                            "return_dict": False,
                            "current_model": self.transformer,
                            "cache_name": "uncond",
                            **extra_model_kwargs,
                        }
                    noise_pred = self.predict_noise_maybe_with_cfg(
                        do_true_cfg=do_true_cfg,
                        true_cfg_scale=guidance_scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=False,
                    )
                    latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                    pbar.update()
            return latents
        finally:
            set_forward_context_denoise_step_idx(None)


class Wan21FLF2VPipelineBase(Wan21I2VPipelineBase):
    support_image_input = True

    def prepare_i2v_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if last_image is None:
            raise ValueError("Wan2.1 FLF2V requires a last image for conditioning.")
        if num_frames < 2:
            raise ValueError("Wan2.1 FLF2V requires at least two frames.")

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (
            batch_size,
            self.vae.config.z_dim,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)
        last_image = last_image.unsqueeze(2)
        video_condition = torch.cat(
            [
                image,
                image.new_zeros(
                    image.shape[0],
                    image.shape[1],
                    num_frames - 2,
                    height,
                    width,
                ),
                last_image,
            ],
            dim=2,
        )
        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)
        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)
        latent_condition = latent_condition.to(dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latent_condition.device, latent_condition.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1,
            self.vae.config.z_dim,
            1,
            1,
            1,
        ).to(latent_condition.device, latent_condition.dtype)
        latent_condition = (latent_condition - latents_mean) * latents_std

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)
        mask_lat_size[:, :, 1 : num_frames - 1] = 0
        first_frame_mask = torch.repeat_interleave(
            mask_lat_size[:, :, 0:1],
            dim=2,
            repeats=self.vae_scale_factor_temporal,
        )
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size,
            -1,
            self.vae_scale_factor_temporal,
            latent_height,
            latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(1, 2).to(latent_condition.device)
        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    @torch.inference_mode()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        (
            prompt_embeds,
            negative_prompt_embeds,
            height,
            width,
            num_frames,
            timesteps,
            generator,
            dtype,
        ) = self._prepare_common_forward(
            req,
            prompt,
            negative_prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            frame_num,
            generator,
            prompt_embeds,
            negative_prompt_embeds,
        )
        multi_modal_data = _multi_modal_data(req.prompts[0]) or {}
        raw_image = multi_modal_data.get("image")
        raw_last_image = multi_modal_data.get("last_image")
        if isinstance(raw_image, list):
            if raw_last_image is not None:
                raise ValueError(
                    "Wan2.1 FLF2V accepts either image as [first, last] or last_image, "
                    "not both."
                )
            if len(raw_image) != 2:
                raise ValueError(
                    "Wan2.1 FLF2V image list must contain exactly two images: "
                    "[first, last]."
                )
            raw_image, raw_last_image = raw_image
        if raw_image is None or raw_last_image is None:
            raise ValueError("Wan2.1 FLF2V requires both first and last images.")

        image = cast(PIL.Image.Image | torch.Tensor, raw_image)
        last_image = cast(PIL.Image.Image | torch.Tensor, raw_last_image)
        if image_embeds is None:
            image_embeds = self.encode_image([image, last_image], self.device)
        image_embeds = _prepare_wan21_flf2v_image_embeds(
            image_embeds,
            prompt_embeds.shape[0],
        ).to(dtype)

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
            image_tensor = video_processor.preprocess(image, height=height, width=width)
        else:
            image_tensor = image
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)

        if isinstance(last_image, PIL.Image.Image):
            last_image = last_image.resize((width, height), PIL.Image.Resampling.LANCZOS)
            last_image_tensor = video_processor.preprocess(
                last_image,
                height=height,
                width=width,
            )
        else:
            last_image_tensor = last_image
        last_image_tensor = last_image_tensor.to(self.device, dtype=torch.float32)

        latents, condition = self.prepare_i2v_latents(
            image=image_tensor,
            batch_size=prompt_embeds.shape[0],
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=req.sampling_params.latents,
            last_image=last_image_tensor,
        )
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=cast(float, self._guidance_scale),
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            extra_model_kwargs={
                "encoder_hidden_states_image": image_embeds,
                "condition": condition,
            },
        )
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        output = self._decode_latents(latents, output_type)
        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )


class Wan21VACEPipelineBase(Wan21PipelineBase, SupportImageInput):
    support_image_input = True

    def _create_transformer(self, config: dict[str, Any]) -> Wan21VACETransformer3DModel:
        quant_config = getattr(self.od_config, "quantization_config", None)
        return create_vace_transformer_from_config(
            config,
            quant_config=quant_config,
            prefix=f"{self.prefix}transformer",
        )

    def preprocess_conditions(
        self,
        video: list | torch.Tensor | None,
        mask: list | torch.Tensor | None,
        reference_images: list[PIL.Image.Image] | None,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]:
        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        if video is None:
            video = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            image_size = (height, width)
        else:
            base = self.vae_scale_factor_spatial * self.transformer_config.patch_size[1]
            if isinstance(video, list):
                video_height, video_width = video_processor.get_default_height_width(video[0])
                if video_height * video_width > height * width:
                    scale = min(width / video_width, height / video_height)
                    video_height = int(video_height * scale)
                    video_width = int(video_width * scale)
                video_height = max(base, (video_height // base) * base)
                video_width = max(base, (video_width // base) * base)
                video = video_processor.preprocess_video(video, video_height, video_width)
            image_size = (video.shape[-2], video.shape[-1])

        if mask is None:
            mask = torch.ones_like(video)
        elif isinstance(mask, list):
            mask = video_processor.preprocess_video(mask, image_size[0], image_size[1])
            mask = torch.clamp((mask + 1) / 2, min=0, max=1)

        video = video.to(dtype=dtype, device=device)
        mask = mask.to(dtype=dtype, device=device)

        ref_images_processed: list[list[torch.Tensor]] = []
        if reference_images:
            preprocessed = []
            for image in reference_images:
                image_tensor = video_processor.preprocess(image, None, None)
                img_h, img_w = image_tensor.shape[-2:]
                scale = min(image_size[0] / img_h, image_size[1] / img_w)
                new_h, new_w = int(img_h * scale), int(img_w * scale)
                resized = torch.nn.functional.interpolate(
                    image_tensor,
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                canvas = torch.ones(3, *image_size, device=device, dtype=dtype)
                top = (image_size[0] - new_h) // 2
                left = (image_size[1] - new_w) // 2
                canvas[:, top : top + new_h, left : left + new_w] = resized
                preprocessed.append(canvas)
            ref_images_processed = [preprocessed]
        else:
            ref_images_processed = [[]]
        return video, mask, ref_images_processed

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        reference_images: list[list[torch.Tensor]],
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        vae_dtype = self.vae.dtype
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=device,
            dtype=torch.float32,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=device,
            dtype=torch.float32,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)

        mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)
        video = video.to(dtype=vae_dtype)
        inactive = video * (1 - mask)
        reactive = video * mask
        inactive_latent = retrieve_latents(
            self.vae.encode(inactive),
            generator,
            sample_mode="argmax",
        )
        reactive_latent = retrieve_latents(
            self.vae.encode(reactive),
            generator,
            sample_mode="argmax",
        )
        inactive_latent = ((inactive_latent.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive_latent = ((reactive_latent.float() - latents_mean) * latents_std).to(vae_dtype)
        latents = torch.cat([inactive_latent, reactive_latent], dim=1)

        latent_list = []
        for latent, ref_batch in zip(latents, reference_images):
            for ref_image in ref_batch:
                ref_image = ref_image.to(dtype=vae_dtype)
                ref_image = ref_image[None, :, None, :, :]
                ref_latent = retrieve_latents(
                    self.vae.encode(ref_image),
                    generator,
                    sample_mode="argmax",
                )
                ref_latent = ((ref_latent.float() - latents_mean) * latents_std).to(vae_dtype)
                ref_latent = ref_latent.squeeze(0)
                ref_latent = torch.cat([ref_latent, torch.zeros_like(ref_latent)], dim=0)
                latent = torch.cat([ref_latent, latent], dim=1)
            latent_list.append(latent)
        return torch.stack(latent_list)

    def prepare_masks(
        self,
        mask: torch.Tensor,
        reference_images: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        transformer_patch_size = self.transformer_config.patch_size[1]
        mask_list = []
        for mask_, ref_batch in zip(mask, reference_images):
            _, num_frames, height, width = mask_.shape
            new_num_frames = (num_frames + self.vae_scale_factor_temporal - 1) // self.vae_scale_factor_temporal
            new_height = (
                height
                // (self.vae_scale_factor_spatial * transformer_patch_size)
                * transformer_patch_size
            )
            new_width = (
                width
                // (self.vae_scale_factor_spatial * transformer_patch_size)
                * transformer_patch_size
            )
            mask_ = mask_[0, :, :, :]
            mask_ = mask_.view(
                num_frames,
                new_height,
                self.vae_scale_factor_spatial,
                new_width,
                self.vae_scale_factor_spatial,
            )
            mask_ = mask_.permute(2, 4, 0, 1, 3).flatten(0, 1)
            mask_ = torch.nn.functional.interpolate(
                mask_.unsqueeze(0),
                size=(new_num_frames, new_height, new_width),
                mode="nearest-exact",
            ).squeeze(0)
            num_ref = len(ref_batch)
            if num_ref > 0:
                mask_padding = torch.zeros_like(mask_[:, :num_ref, :, :])
                mask_ = torch.cat([mask_padding, mask_], dim=1)
            mask_list.append(mask_)
        return torch.stack(mask_list)

    def _normalise_conditioning_scale(
        self,
        conditioning_scale: float | list[float] | torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        vace_layers = self.transformer.config.vace_layers
        if isinstance(conditioning_scale, (int, float)):
            conditioning_scale = [float(conditioning_scale)] * len(vace_layers)
        if isinstance(conditioning_scale, list):
            if len(conditioning_scale) != len(vace_layers):
                raise ValueError(
                    f"Length of conditioning_scale {len(conditioning_scale)} "
                    f"does not match number of VACE layers {len(vace_layers)}."
                )
            conditioning_scale = torch.tensor(conditioning_scale)
        if conditioning_scale.size(0) != len(vace_layers):
            raise ValueError(
                f"Length of conditioning_scale {conditioning_scale.size(0)} "
                f"does not match number of VACE layers {len(vace_layers)}."
            )
        return conditioning_scale.to(device=self.device, dtype=dtype)

    def _zero_vace_condition(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        channels = self.transformer.config.vace_in_channels
        return torch.zeros(
            batch_size,
            channels,
            num_latent_frames,
            latent_height,
            latent_width,
            device=self.device,
            dtype=dtype,
        )

    @torch.inference_mode()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        conditioning_scale: float | list[float] | torch.Tensor = 1.0,
        **kwargs,
    ) -> DiffusionOutput:
        (
            prompt_embeds,
            negative_prompt_embeds,
            height,
            width,
            num_frames,
            timesteps,
            generator,
            dtype,
        ) = self._prepare_common_forward(
            req,
            prompt,
            negative_prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            frame_num,
            generator,
            prompt_embeds,
            negative_prompt_embeds,
        )
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        conditioning_scale = extra_args.get("conditioning_scale", conditioning_scale)
        conditioning_scale = self._normalise_conditioning_scale(
            conditioning_scale,
            dtype=dtype,
        )
        first_prompt = _ensure_prompt_dict(req.prompts[0])
        additional_information = first_prompt.get("additional_information", {}) or {}
        reference_images = additional_information.get("reference_images")
        source_video = additional_information.get("source_video")
        source_mask = additional_information.get("mask")
        video, mask, reference_images_processed = self.preprocess_conditions(
            video=source_video,
            mask=source_mask,
            reference_images=reference_images,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.device,
        )
        if isinstance(generator, list):
            raise ValueError("Wan2.1 VACE does not support a list of generators.")
        conditioning_latents = self.prepare_video_latents(
            video,
            mask,
            reference_images_processed,
            generator,
            self.device,
        )
        mask_encoded = self.prepare_masks(mask, reference_images_processed)
        conditioning_latents = torch.cat([conditioning_latents, mask_encoded], dim=1)
        conditioning_latents = conditioning_latents.to(dtype)
        num_reference_images = len(reference_images_processed[0])
        noise_num_frames = num_frames + num_reference_images * self.vae_scale_factor_temporal
        latents = self.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=self.transformer_config.in_channels,
            height=height,
            width=width,
            num_frames=noise_num_frames,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=req.sampling_params.latents,
        )
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=cast(float, self._guidance_scale),
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            extra_model_kwargs={
                "control_hidden_states": conditioning_latents,
                "control_hidden_states_scale": conditioning_scale,
            },
        )
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None
        if output_type != "latent" and num_reference_images > 0:
            latents = latents[:, :, num_reference_images:]
        output = self._decode_latents(latents, output_type)
        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )
