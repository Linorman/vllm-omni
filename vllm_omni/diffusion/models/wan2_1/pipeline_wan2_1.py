# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm_omni.diffusion.models.wan2_1.pipeline_wan2_1_common import (
    Wan21PipelineBase,
    get_wan21_post_process_func,
    get_wan21_pre_process_func,
)


class Wan21Pipeline(Wan21PipelineBase):
    """Wan2.1 text-to-video pipeline using vLLM-Omni's native denoising path."""


__all__ = [
    "Wan21Pipeline",
    "get_wan21_post_process_func",
    "get_wan21_pre_process_func",
]
