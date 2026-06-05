# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.diffusion.models.wan2_1.conftest import StubTransformer, StubVAE, noop_progress_bar

pytest.importorskip("vllm")

from vllm_omni.diffusion.models.wan2_1.pipeline_wan2_1_common import Wan21FLF2VPipelineBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline() -> Wan21FLF2VPipelineBase:
    pipeline = object.__new__(Wan21FLF2VPipelineBase)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer(in_channels=12, out_channels=4)
    pipeline.transformer_config = pipeline.transformer.config
    pipeline.vae = StubVAE(z_dim=4)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.progress_bar = noop_progress_bar
    return pipeline


def test_flf2v_prepare_latents_requires_last_image() -> None:
    pipeline = _make_pipeline()

    with pytest.raises(ValueError, match="requires a last image"):
        pipeline.prepare_i2v_latents(
            image=torch.zeros(1, 3, 16, 16),
            batch_size=1,
            height=16,
            width=16,
            num_frames=5,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=None,
        )


def test_flf2v_prepare_latents_marks_first_and_last_frames_as_known() -> None:
    pipeline = _make_pipeline()
    latents, condition = pipeline.prepare_i2v_latents(
        image=torch.zeros(1, 3, 16, 16),
        last_image=torch.ones(1, 3, 16, 16),
        batch_size=1,
        height=16,
        width=16,
        num_frames=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(0),
    )

    assert latents.shape == (1, 4, 2, 2, 2)
    assert condition.shape == (1, 8, 2, 2, 2)
    mask = condition[:, :4]
    assert mask[:, :, 0].sum() > 0
    assert mask[:, :, 1].sum() > 0
