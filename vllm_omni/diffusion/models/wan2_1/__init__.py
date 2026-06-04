# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from importlib import import_module
from typing import Any

from .model_index import (
    WAN21_I2V_PIPELINE,
    WAN21_T2V_PIPELINE,
    WAN21_VACE_PIPELINE,
    resolve_wan21_pipeline_class_name,
)

_LAZY_EXPORTS = {
    "Wan21Pipeline": ("pipeline_wan2_1", "Wan21Pipeline"),
    "get_wan21_post_process_func": (
        "pipeline_wan2_1",
        "get_wan21_post_process_func",
    ),
    "get_wan21_pre_process_func": (
        "pipeline_wan2_1",
        "get_wan21_pre_process_func",
    ),
    "Wan21I2VPipeline": ("pipeline_wan2_1_i2v", "Wan21I2VPipeline"),
    "get_wan21_i2v_post_process_func": (
        "pipeline_wan2_1_i2v",
        "get_wan21_i2v_post_process_func",
    ),
    "get_wan21_i2v_pre_process_func": (
        "pipeline_wan2_1_i2v",
        "get_wan21_i2v_pre_process_func",
    ),
    "Wan21VACEPipeline": ("pipeline_wan2_1_vace", "Wan21VACEPipeline"),
    "get_wan21_vace_post_process_func": (
        "pipeline_wan2_1_vace",
        "get_wan21_vace_post_process_func",
    ),
    "get_wan21_vace_pre_process_func": (
        "pipeline_wan2_1_vace",
        "get_wan21_vace_pre_process_func",
    ),
    "Wan21Transformer3DModel": ("wan2_1_transformer", "Wan21Transformer3DModel"),
    "Wan21VACETransformer3DModel": (
        "wan2_1_vace_transformer",
        "Wan21VACETransformer3DModel",
    ),
}

__all__ = [
    "WAN21_T2V_PIPELINE",
    "WAN21_I2V_PIPELINE",
    "WAN21_VACE_PIPELINE",
    "resolve_wan21_pipeline_class_name",
    "Wan21Pipeline",
    "get_wan21_post_process_func",
    "get_wan21_pre_process_func",
    "Wan21I2VPipeline",
    "get_wan21_i2v_post_process_func",
    "get_wan21_i2v_pre_process_func",
    "Wan21VACEPipeline",
    "get_wan21_vace_post_process_func",
    "get_wan21_vace_pre_process_func",
    "Wan21Transformer3DModel",
    "Wan21VACETransformer3DModel",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value
