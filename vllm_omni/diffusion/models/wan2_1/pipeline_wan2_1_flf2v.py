# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm_omni.diffusion.models.wan2_1.pipeline_wan2_1_common import (
    Wan21FLF2VPipelineBase,
    get_wan21_flf2v_post_process_func,
    get_wan21_flf2v_pre_process_func,
)


class Wan21FLF2VPipeline(Wan21FLF2VPipelineBase):
    """Wan2.1 first-last-frame-to-video pipeline using the native denoising path."""


__all__ = [
    "Wan21FLF2VPipeline",
    "get_wan21_flf2v_post_process_func",
    "get_wan21_flf2v_pre_process_func",
]
