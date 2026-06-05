# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.wan2_1.conftest import StubTransformer, StubVAE, noop_progress_bar

pytest.importorskip("vllm")

from vllm_omni.diffusion.models.wan2_1.pipeline_wan2_1_common import (
    Wan21VACEPipelineBase,
    get_wan21_vace_pre_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline() -> Wan21VACEPipelineBase:
    pipeline = object.__new__(Wan21VACEPipelineBase)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer(in_channels=4, out_channels=4)
    pipeline.transformer_config = pipeline.transformer.config
    pipeline.vae = StubVAE(z_dim=4)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.progress_bar = noop_progress_bar
    return pipeline


def test_vace_preprocess_conditions_builds_default_video_mask_and_reference_image() -> None:
    pipeline = _make_pipeline()
    reference = Image.new("RGB", (8, 8), color=(255, 0, 0))

    video, mask, references = pipeline.preprocess_conditions(
        video=None,
        mask=None,
        reference_images=[reference],
        height=16,
        width=16,
        num_frames=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert video.shape == (1, 3, 5, 16, 16)
    assert mask.shape == (1, 3, 5, 16, 16)
    assert len(references) == 1
    assert len(references[0]) == 1
    assert references[0][0].shape == (3, 16, 16)
    torch.testing.assert_close(mask, torch.ones_like(mask))


def test_vace_pre_process_collects_reference_video_and_mask_inputs() -> None:
    image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    mask = Image.new("L", (8, 8), color=255)
    request = type(
        "Request",
        (),
        {
            "prompts": [
                {
                    "prompt": "edit the source video",
                    "multi_modal_data": {
                        "image": image,
                        "video": [image],
                        "mask": [mask],
                    },
                }
            ]
        },
    )()

    processed = get_wan21_vace_pre_process_func(object())(request)

    additional_information = processed.prompts[0]["additional_information"]
    assert len(additional_information["reference_images"]) == 1
    assert len(additional_information["source_video"]) == 1
    assert len(additional_information["mask"]) == 1
    assert additional_information["reference_images"][0].mode == "RGB"
    assert additional_information["source_video"][0].mode == "RGB"
    assert additional_information["mask"][0].mode == "L"


def test_vace_prepare_masks_pads_reference_image_slots() -> None:
    pipeline = _make_pipeline()

    masks = pipeline.prepare_masks(
        mask=torch.ones(1, 3, 5, 16, 16),
        reference_images=[[torch.zeros(3, 16, 16)]],
    )

    assert masks.shape == (1, 64, 3, 2, 2)
    torch.testing.assert_close(masks[:, :, 0], torch.zeros_like(masks[:, :, 0]))


def test_vace_conditioning_scale_expands_to_vace_layers() -> None:
    pipeline = _make_pipeline()

    scale = pipeline._normalise_conditioning_scale(0.5, torch.float32)

    torch.testing.assert_close(scale, torch.tensor([0.5, 0.5, 0.5]))


def test_vace_conditioning_scale_rejects_wrong_layer_count() -> None:
    pipeline = _make_pipeline()

    with pytest.raises(ValueError, match="does not match number of VACE layers"):
        pipeline._normalise_conditioning_scale([1.0, 0.5], torch.float32)


def test_vace_diffuse_passes_control_kwargs_to_cfg_branches() -> None:
    pipeline = _make_pipeline()
    latents = torch.zeros(1, 4, 1, 2, 2)
    control = torch.ones(1, 12, 1, 2, 2)
    control_scale = torch.tensor([1.0, 0.5, 0.25])
    calls = []

    def fake_predict_noise_maybe_with_cfg(**kwargs):
        calls.append(kwargs)
        return torch.ones_like(latents)

    pipeline.predict_noise_maybe_with_cfg = fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, current, cfg: current + noise  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([3]),
        prompt_embeds=torch.zeros(1, 2, 3),
        negative_prompt_embeds=torch.zeros(1, 2, 3),
        guidance_scale=4.0,
        dtype=torch.float32,
        attention_kwargs={},
        extra_model_kwargs={
            "control_hidden_states": control,
            "control_hidden_states_scale": control_scale,
        },
    )

    positive = calls[0]["positive_kwargs"]
    negative = calls[0]["negative_kwargs"]
    assert calls[0]["do_true_cfg"] is True
    torch.testing.assert_close(positive["control_hidden_states"], control)
    torch.testing.assert_close(negative["control_hidden_states_scale"], control_scale)
    torch.testing.assert_close(result, torch.ones_like(latents))
