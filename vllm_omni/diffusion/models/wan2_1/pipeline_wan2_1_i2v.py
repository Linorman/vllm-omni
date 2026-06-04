# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm_omni.diffusion.models.wan2_1.pipeline_wan2_1_common import (
    Wan21I2VPipelineBase,
    get_wan21_i2v_post_process_func,
    get_wan21_i2v_pre_process_func,
)


class Wan21I2VPipeline(Wan21I2VPipelineBase):
    """Wan2.1 image-to-video pipeline using vLLM-Omni's native denoising path."""


__all__ = [
    "Wan21I2VPipeline",
    "get_wan21_i2v_post_process_func",
    "get_wan21_i2v_pre_process_func",
]
