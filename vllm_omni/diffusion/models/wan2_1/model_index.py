# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

WAN21_T2V_PIPELINE = "Wan21Pipeline"
WAN21_I2V_PIPELINE = "Wan21I2VPipeline"
WAN21_VACE_PIPELINE = "Wan21VACEPipeline"

_WAN21_CLASS_MAP = {
    "WanPipeline": WAN21_T2V_PIPELINE,
    "WanImageToVideoPipeline": WAN21_I2V_PIPELINE,
    "WanVACEPipeline": WAN21_VACE_PIPELINE,
}


def _component_is_present(model_index: Mapping[str, Any], key: str) -> bool:
    value = model_index.get(key)
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) >= 1 and value[0] is not None
    return True


def _looks_like_wan21_model_name(model: str | None) -> bool:
    if not model:
        return False
    normalized = str(model).replace("\\", "/").lower()
    basename = os.path.basename(normalized.rstrip("/"))
    return "wan2.1" in normalized or "wan21" in basename


def _diffusers_version_major_minor(model_index: Mapping[str, Any]) -> tuple[int, int] | None:
    raw = model_index.get("_diffusers_version")
    if not isinstance(raw, str):
        return None
    prefix = raw.split(".dev", 1)[0]
    pieces = prefix.split(".")
    if len(pieces) < 2:
        return None
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError:
        return None


def _has_wan21_diffusers_version(model_index: Mapping[str, Any]) -> bool:
    version = _diffusers_version_major_minor(model_index)
    return version is not None and version < (0, 35)


def _is_wan21_t2v(
    model: str | None,
    model_index: Mapping[str, Any],
    transformer_config: Mapping[str, Any] | None,
) -> bool:
    if model_index.get("_class_name") != "WanPipeline":
        return False
    if _component_is_present(model_index, "transformer_2"):
        return False
    if bool(model_index.get("expand_timesteps", False)):
        return False
    if _looks_like_wan21_model_name(model) or _has_wan21_diffusers_version(model_index):
        return True
    if transformer_config is None:
        return False
    return (
        transformer_config.get("_class_name") == "WanTransformer3DModel"
        and transformer_config.get("in_channels") == 16
        and transformer_config.get("out_channels") == 16
        and transformer_config.get("image_dim") is None
    )


def _is_wan21_i2v(model: str | None, model_index: Mapping[str, Any]) -> bool:
    if model_index.get("_class_name") != "WanImageToVideoPipeline":
        return False
    if _component_is_present(model_index, "transformer_2"):
        return False
    if _component_is_present(model_index, "image_encoder"):
        return True
    return _looks_like_wan21_model_name(model) or _has_wan21_diffusers_version(model_index)


def _is_wan21_vace(
    model: str | None,
    model_index: Mapping[str, Any],
    transformer_config: Mapping[str, Any] | None,
) -> bool:
    if model_index.get("_class_name") != "WanVACEPipeline":
        return False
    if _component_is_present(model_index, "transformer_2"):
        return False
    if _looks_like_wan21_model_name(model) or _has_wan21_diffusers_version(model_index):
        return True
    if transformer_config is None:
        return False
    return transformer_config.get("_class_name") == "WanVACETransformer3DModel"


def resolve_wan21_pipeline_class_name(
    model: str | None,
    model_index: Mapping[str, Any] | None,
    transformer_config: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the internal Wan2.1 pipeline class for a diffusers model."""
    if not model_index:
        return None
    class_name = model_index.get("_class_name")
    if class_name not in _WAN21_CLASS_MAP:
        return None
    if _is_wan21_t2v(model, model_index, transformer_config):
        return WAN21_T2V_PIPELINE
    if _is_wan21_i2v(model, model_index):
        return WAN21_I2V_PIPELINE
    if _is_wan21_vace(model, model_index, transformer_config):
        return WAN21_VACE_PIPELINE
    return None

__all__ = [
    "WAN21_T2V_PIPELINE",
    "WAN21_I2V_PIPELINE",
    "WAN21_VACE_PIPELINE",
    "resolve_wan21_pipeline_class_name",
]
