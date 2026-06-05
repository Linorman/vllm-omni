# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.diffusion.models.wan2_1.conftest import StubTransformer, StubVAE, noop_progress_bar

pytest.importorskip("vllm")

from vllm_omni.diffusion.models.wan2_1.pipeline_wan2_1_common import Wan21I2VPipelineBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline() -> Wan21I2VPipelineBase:
    pipeline = object.__new__(Wan21I2VPipelineBase)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer(in_channels=12, out_channels=4)
    pipeline.transformer_config = pipeline.transformer.config
    pipeline.vae = StubVAE(z_dim=4)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.progress_bar = noop_progress_bar
    return pipeline


def test_i2v_prepare_latents_builds_first_frame_condition() -> None:
    pipeline = _make_pipeline()
    latents, condition = pipeline.prepare_i2v_latents(
        image=torch.zeros(1, 3, 16, 16),
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
    torch.testing.assert_close(condition[:, :4, 0], torch.ones(1, 4, 2, 2))
    torch.testing.assert_close(condition[:, 4:, 0], torch.ones(1, 4, 2, 2))


def test_i2v_diffuse_concatenates_latents_and_condition() -> None:
    pipeline = _make_pipeline()
    latents = torch.zeros(1, 4, 1, 2, 2)
    condition = torch.ones(1, 8, 1, 2, 2)
    calls = []

    def fake_predict_noise_maybe_with_cfg(**kwargs):
        calls.append(kwargs)
        return torch.ones_like(latents)

    pipeline.predict_noise_maybe_with_cfg = fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, current, cfg: current + noise  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([9]),
        prompt_embeds=torch.zeros(1, 2, 3),
        negative_prompt_embeds=None,
        guidance_scale=1.0,
        dtype=torch.float32,
        attention_kwargs={},
        extra_model_kwargs={
            "condition": condition,
            "encoder_hidden_states_image": torch.zeros(1, 2, 3),
        },
    )

    hidden_states = calls[0]["positive_kwargs"]["hidden_states"]
    assert hidden_states.shape == (1, 12, 1, 2, 2)
    torch.testing.assert_close(hidden_states[:, :4], torch.zeros_like(latents))
    torch.testing.assert_close(hidden_states[:, 4:], torch.ones_like(condition))
    torch.testing.assert_close(result, torch.ones_like(latents))
