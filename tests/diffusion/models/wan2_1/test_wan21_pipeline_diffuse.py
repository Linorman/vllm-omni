# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.diffusion.models.wan2_1.conftest import StubTransformer, StubVAE, noop_progress_bar

pytest.importorskip("vllm")

from vllm_omni.diffusion.models.wan2_1.pipeline_wan2_1_common import Wan21PipelineBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline() -> Wan21PipelineBase:
    pipeline = object.__new__(Wan21PipelineBase)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer(in_channels=4, out_channels=4)
    pipeline.transformer_config = pipeline.transformer.config
    pipeline.vae = StubVAE(z_dim=4)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.progress_bar = noop_progress_bar
    return pipeline


def test_t2v_diffuse_builds_cfg_positive_and_negative_branches() -> None:
    pipeline = _make_pipeline()
    latents = torch.zeros(1, 4, 1, 2, 2)
    calls = []

    def fake_predict_noise_maybe_with_cfg(**kwargs):
        calls.append(kwargs)
        return torch.ones_like(latents)

    pipeline.predict_noise_maybe_with_cfg = fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, current, cfg: current + noise  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([5]),
        prompt_embeds=torch.zeros(1, 2, 3),
        negative_prompt_embeds=torch.zeros(1, 2, 3),
        guidance_scale=5.0,
        dtype=torch.float32,
        attention_kwargs={"scale": 1.0},
    )

    assert calls[0]["do_true_cfg"] is True
    assert calls[0]["true_cfg_scale"] == 5.0
    assert calls[0]["positive_kwargs"]["cache_name"] == "cond"
    assert calls[0]["negative_kwargs"]["cache_name"] == "uncond"
    assert calls[0]["positive_kwargs"]["attention_kwargs"] == {"scale": 1.0}
    torch.testing.assert_close(result, torch.ones_like(latents))
