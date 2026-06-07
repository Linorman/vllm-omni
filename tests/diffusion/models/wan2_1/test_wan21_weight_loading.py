# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WAN21_DIR = PROJECT_ROOT / "vllm_omni" / "diffusion" / "models" / "wan2_1"
WAN21_TRANSFORMER = WAN21_DIR / "wan2_1_transformer.py"
WAN21_VACE_TRANSFORMER = WAN21_DIR / "wan2_1_vace_transformer.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_source(path: Path, name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{name} not found in {path}")


def test_wan21_load_weights_packs_self_attention_qkv():
    source = _function_source(WAN21_TRANSFORMER, "load_weights")
    assert "stacked_params_mapping" in source
    assert '".attn1.to_qkv"' in source
    assert '".attn1.to_q"' in source
    assert '".attn1.to_k"' in source
    assert '".attn1.to_v"' in source
    assert '"q"' in source
    assert '"k"' in source
    assert '"v"' in source
    assert "param.weight_loader" in source


def test_wan21_load_weights_remaps_parallel_ffn_and_output_projection_names():
    source = _function_source(WAN21_TRANSFORMER, "load_weights")
    assert '".ffn.net.0."' in source
    assert '".ffn.net_0."' in source
    assert '".ffn.net.2."' in source
    assert '".ffn.net_2."' in source
    assert '".to_out.0."' in source
    assert '".to_out."' in source


def test_wan21_load_weights_handles_scale_shift_and_tp_norm_shards():
    source = _function_source(WAN21_TRANSFORMER, "load_weights")
    assert '"scale_shift_table": "output_scale_shift_prepare.scale_shift_table"' in source
    assert "get_tensor_model_parallel_rank()" in source
    assert "get_tensor_model_parallel_world_size()" in source
    assert "shard_size = loaded_weight.shape[0] // tp_size" in source
    assert ".attn1.norm_q." in source
    assert ".attn1.norm_k." in source


def test_wan21_vace_transformer_inherits_parent_weight_loader_and_maps_vace_blocks():
    source = _source(WAN21_VACE_TRANSFORMER)
    assert "Wan21Transformer3DModel" in source
    assert "VaceWan21TransformerBlock" in source
    assert 'prefix=f"{prefix}.vace_blocks.{i}" if prefix else f"vace_blocks.{i}"' in source
    assert "proj_in" in source
    assert "proj_out" in source
