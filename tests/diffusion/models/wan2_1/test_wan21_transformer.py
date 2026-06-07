# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
from pathlib import Path

import torch

from vllm_omni.diffusion.models.wan2_1.wan2_1_transformer import WanImageEmbedding, WanRotaryPosEmbed


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WAN21_DIR = PROJECT_ROOT / "vllm_omni" / "diffusion" / "models" / "wan2_1"
WAN21_TRANSFORMER = WAN21_DIR / "wan2_1_transformer.py"
WAN21_VACE_TRANSFORMER = WAN21_DIR / "wan2_1_vace_transformer.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(_source(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _class_method_source(source: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef) and statement.name == method_name:
                return ast.get_source_segment(source, statement) or ""
    raise AssertionError(f"{class_name}.{method_name} not found")


def test_wan21_native_transformer_declares_runtime_metadata():
    source = _source(WAN21_TRANSFORMER)
    assert "class Wan21Transformer3DModel" in source
    assert "_layerwise_offload_blocks_attrs" in source
    assert "packed_modules_mapping" in source
    assert "_hsdp_shard_conditions" in source
    assert "_sp_plan" in source
    assert "_repeated_blocks" in source


def test_wan21_native_transformer_uses_vllm_parallel_layers():
    source = _source(WAN21_TRANSFORMER)
    assert "QKVParallelLinear" in source
    assert "ColumnParallelLinear" in source
    assert "RowParallelLinear" in source
    assert "Conv3dLayer" in source
    assert "quant_config" in source


def test_wan21_sp_plan_has_shard_and_gather_points():
    source = _source(WAN21_TRANSFORMER)
    assert '"rope"' in source
    assert '"blocks.0"' in source
    assert '"proj_out"' in source
    assert "SequenceParallelInput" in source
    assert "SequenceParallelOutput" in source
    assert "auto_pad=True" in source


def test_wan21_hsdp_condition_is_conservative():
    source = _source(WAN21_TRANSFORMER)
    assert "def _is_transformer_block(" in source
    assert 'name.startswith("blocks.")' in source
    assert 'name.split(".")[-1].isdigit()' in source


def test_wan21_vace_transformer_declares_vace_metadata_and_shard_point():
    source = _source(WAN21_VACE_TRANSFORMER)
    assert "class Wan21VACETransformer3DModel" in source
    assert "class VaceWan21TransformerBlock" in source
    assert "vace_blocks" in source
    assert "vace_patch_embedding" in source
    assert "_sp_shard_point" in source
    assert "SequenceParallelInput" in source


def test_wan21_vace_quant_prefix_reaches_vace_blocks():
    source = _source(WAN21_VACE_TRANSFORMER)
    assert 'prefix: str = ""' in source
    assert 'prefix=f"{prefix}.vace_blocks.{i}" if prefix else f"vace_blocks.{i}"' in source


def test_wan21_transformers_do_not_import_wan22_modules():
    for path in (WAN21_TRANSFORMER, WAN21_VACE_TRANSFORMER):
        assert not any("wan2_2" in module for module in _imports_from(path)), path


def test_wan21_block_modulation_uses_diffusers_fp32_order():
    source = _source(WAN21_TRANSFORMER)
    block_forward = _class_method_source(source, "WanTransformerBlock", "forward")

    assert "temb.float()" in block_forward
    assert "self.norm1(hidden_states.float(), scale_msa, shift_msa)" in block_forward
    assert "hidden_states.float() + attn_output * gate_msa" in block_forward
    assert "self.norm2(hidden_states.float()).type_as(hidden_states)" in block_forward
    assert "self.norm3(hidden_states.float(), c_scale_msa, c_shift_msa)" in block_forward
    assert "hidden_states.float() + ff_output.float() * c_gate_msa" in block_forward


def test_wan21_output_norm_uses_diffusers_fp32_order():
    source = _source(WAN21_TRANSFORMER)
    forward_source = _class_method_source(source, "Wan21Transformer3DModel", "forward")

    assert "self.norm_out(hidden_states.float(), scale, shift).type_as(hidden_states)" in forward_source


def test_wan21_vace_output_norm_uses_diffusers_fp32_order():
    source = _source(WAN21_VACE_TRANSFORMER)
    forward_source = _class_method_source(source, "Wan21VACETransformer3DModel", "forward")

    assert "self.norm_out(hidden_states.float(), scale, shift).type_as(hidden_states)" in forward_source


def test_wan21_vace_context_embedding_materializes_diffusers_linear_layout():
    source = _source(WAN21_VACE_TRANSFORMER)
    embed_source = _class_method_source(source, "Wan21VACETransformer3DModel", "embed_vace_context")

    assert "vace_embeds = vace_embeds.contiguous()" in embed_source


def test_wan21_rotary_embeddings_use_current_rope_buffer_dtype():
    for path, class_name in (
        (WAN21_TRANSFORMER, "Wan21Transformer3DModel"),
        (WAN21_VACE_TRANSFORMER, "Wan21VACETransformer3DModel"),
    ):
        forward_source = _class_method_source(_source(path), class_name, "forward")

        assert "freqs_cos[..., 0::2]" in forward_source
        assert "freqs_sin[..., 1::2]" in forward_source
        assert "freqs_cos[..., 0::2].to(hidden_states.dtype)" not in forward_source
        assert "freqs_sin[..., 1::2].to(hidden_states.dtype)" not in forward_source


def test_wan21_self_attention_uses_diffusers_style_native_rotary_path():
    source = _source(WAN21_TRANSFORMER)
    forward_source = _class_method_source(source, "WanSelfAttention", "forward")

    assert "force_native=True" in forward_source


def test_wan21_rotary_pos_embed_preserves_module_cast_dtype_like_diffusers():
    source = _source(WAN21_TRANSFORMER)
    forward_source = _class_method_source(source, "WanRotaryPosEmbed", "forward")

    assert "freqs_cos.to(device=hidden_states.device)" in forward_source
    assert "freqs_sin.to(device=hidden_states.device)" in forward_source
    assert "dtype=torch.float32" not in forward_source


def test_wan21_rotary_pos_embed_matches_diffusers_float64_frequency_grid():
    source = _source(WAN21_TRANSFORMER)
    build_source = _class_method_source(source, "WanRotaryPosEmbed", "_build_freqs")

    assert "freqs_dtype = torch.float64" in build_source
    assert "current_omni_platform.supports_float64()" not in build_source


def test_wan21_rotary_pos_embed_follows_module_to_dtype_like_diffusers():
    rope = WanRotaryPosEmbed(attention_head_dim=128, patch_size=(1, 2, 2), max_seq_len=32)
    expected_cos = rope.freqs_cos.clone()
    expected_sin = rope.freqs_sin.clone()

    rope.to(dtype=torch.bfloat16)

    assert rope.freqs_cos.dtype is torch.bfloat16
    assert rope.freqs_sin.dtype is torch.bfloat16
    torch.testing.assert_close(rope.freqs_cos, expected_cos.to(torch.bfloat16), atol=0, rtol=0)
    torch.testing.assert_close(rope.freqs_sin, expected_sin.to(torch.bfloat16), atol=0, rtol=0)


def test_wan21_rotary_pos_embed_uses_default_dtype_at_construction_like_loader():
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        rope = WanRotaryPosEmbed(attention_head_dim=128, patch_size=(1, 2, 2), max_seq_len=32)
    finally:
        torch.set_default_dtype(old_dtype)

    assert rope.freqs_cos.dtype is torch.bfloat16
    assert rope.freqs_sin.dtype is torch.bfloat16


def test_wan21_image_embedding_norms_use_diffusers_fp32_order():
    source = _source(WAN21_TRANSFORMER)
    forward_source = _class_method_source(source, "WanImageEmbedding", "forward")

    assert "self.norm1(encoder_hidden_states_image.float())" in forward_source
    assert ".type_as(\n            encoder_hidden_states_image\n        )" in forward_source
    assert "self.norm2(hidden_states.float()).type_as(hidden_states)" in forward_source


def test_wan21_image_embedding_layernorm_eps_matches_diffusers_default():
    image_embedding = WanImageEmbedding(in_features=8, out_features=16)

    assert image_embedding.norm1.eps == 1e-5
    assert image_embedding.norm2.eps == 1e-5


def test_wan21_transformer_follows_reference_pipeline_module_dtype_cast():
    source = _source(WAN21_TRANSFORMER)
    vace_source = _source(WAN21_VACE_TRANSFORMER)

    assert "_keep_diffusers_fp32_modules" not in source
    assert "_keep_diffusers_fp32_modules" not in vace_source
    assert "time_embedder.float()" not in source
    assert "scale_shift_table.data.float()" not in source
    assert "output_scale_shift_prepare.float()" not in source


def test_wan21_patch_embedding_uses_conv_path_for_diffusers_equivalence():
    source = _source(WAN21_TRANSFORMER)

    assert "self.patch_embedding.enable_linear = False" in source


def test_wan21_self_attention_projects_qkv_separately_for_unquantized_models():
    source = _source(WAN21_TRANSFORMER)
    init_source = _class_method_source(source, "WanSelfAttention", "__init__")
    forward_source = _class_method_source(source, "WanSelfAttention", "forward")

    assert "self.use_separate_qkv = quant_config is None" in init_source
    assert "self.use_separate_qkv = quant_config is None and dim >= 5120" not in init_source
    assert "F.linear(hidden_states, weight[:q_size], bias[:q_size]" in forward_source
    assert "torch.cat([query, key, value], dim=-1)" in forward_source


def test_wan21_single_rank_attention_norms_match_diffusers_torch_rmsnorm():
    source = _source(WAN21_TRANSFORMER)
    self_init = _class_method_source(source, "WanSelfAttention", "__init__")
    cross_init = _class_method_source(source, "WanCrossAttention", "__init__")

    assert "from vllm_omni.diffusion.layers.norm import LayerNorm, RMSNorm" not in source
    assert "self.norm_q = nn.RMSNorm(self.tp_inner_dim, eps=eps)" in self_init
    assert "self.norm_k = nn.RMSNorm(self.tp_inner_dim, eps=eps)" in self_init
    assert "self.norm_q = nn.RMSNorm(self.tp_inner_dim, eps=eps)" in cross_init
    assert "self.norm_k = nn.RMSNorm(self.tp_inner_dim, eps=eps)" in cross_init
    assert "self.norm_added_k = nn.RMSNorm(self.tp_inner_dim, eps=eps)" in cross_init
