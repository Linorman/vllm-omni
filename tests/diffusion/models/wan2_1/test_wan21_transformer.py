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


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(_source(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


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
