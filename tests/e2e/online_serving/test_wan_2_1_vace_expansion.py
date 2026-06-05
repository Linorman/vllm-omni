# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GPU acceptance coverage for Wan2.1-VACE native transformer features.

Coverage:
  Single GPU:
    - Cache-DiT + layerwise CPU offload
  Two GPUs:
    - Ulysses-SP = 2
    - Ring = 2
    - CFG-Parallel = 2
    - Pipeline-Parallel = 2
    - TP = 2 + VAE-Patch-Parallel = 2
    - HSDP = 2 + VAE-Patch-Parallel = 2
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

WAN21_VACE_MODELS = [
    ("Wan-AI/Wan2.1-VACE-1.3B-diffusers", "vace_13b"),
    ("Wan-AI/Wan2.1-VACE-14B-diffusers", "vace_14b"),
]
PROMPT = "A cat walking slowly across a sunlit garden path"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

FEATURE_CONFIGS = [
    (
        "cache_dit_layerwise_offload",
        ["--cache-backend", "cache_dit", "--enable-layerwise-offload"],
        SINGLE_CARD_FEATURE_MARKS,
    ),
    ("ulysses_sp", ["--usp", "2"], PARALLEL_FEATURE_MARKS),
    ("ring", ["--ring", "2"], PARALLEL_FEATURE_MARKS),
    ("cfg_parallel", ["--cfg-parallel-size", "2"], PARALLEL_FEATURE_MARKS),
    (
        "tensor_parallel_vae_patch",
        [
            "--tensor-parallel-size",
            "2",
            "--vae-patch-parallel-size",
            "2",
            "--vae-use-tiling",
        ],
        PARALLEL_FEATURE_MARKS,
    ),
    (
        "hsdp_vae_patch",
        [
            "--use-hsdp",
            "--hsdp-shard-size",
            "2",
            "--vae-patch-parallel-size",
            "2",
            "--vae-use-tiling",
        ],
        PARALLEL_FEATURE_MARKS,
    ),
    ("pipeline_parallel", ["--pipeline-parallel-size", "2"], PARALLEL_FEATURE_MARKS),
]


def _get_vace_feature_cases():
    cases = []
    for model_path, model_key in WAN21_VACE_MODELS:
        for feature_id, server_args, marks in FEATURE_CONFIGS:
            cases.append(
                pytest.param(
                    OmniServerParams(model=model_path, server_args=server_args),
                    id=f"{model_key}_{feature_id}",
                    marks=marks,
                )
            )
    return cases


def _synthetic_image_reference() -> str:
    return f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"


@pytest.mark.parametrize(
    "omni_server",
    _get_vace_feature_cases(),
    indirect=True,
)
def test_wan_2_1_vace(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test VACE T2V generation with all supported diffusion acceleration features."""
    result = openai_client.send_video_diffusion_request(
        {
            "model": omni_server.model,
            "form_data": {
                "prompt": PROMPT,
                "height": 480,
                "width": 320,
                "num_frames": 5,
                "fps": 8,
                "num_inference_steps": 2,
                "guidance_scale": 5.0,
                "seed": 42,
            },
        }
    )[0]
    assert result.videos is not None
    assert len(result.videos) == 1
    assert len(result.videos[0]) > 1024


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
                server_args=["--enforce-eager"],
            ),
            id="vace_13b_reference_image",
            marks=SINGLE_CARD_FEATURE_MARKS,
        )
    ],
    indirect=True,
)
def test_wan_2_1_vace_reference_image(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    result = openai_client.send_video_diffusion_request(
        {
            "model": omni_server.model,
            "form_data": {
                "prompt": PROMPT,
                "height": 480,
                "width": 320,
                "num_frames": 5,
                "fps": 8,
                "num_inference_steps": 2,
                "guidance_scale": 5.0,
                "seed": 42,
            },
            "image_reference": _synthetic_image_reference(),
        }
    )[0]
    assert result.videos is not None
    assert len(result.videos) == 1
    assert len(result.videos[0]) > 1024
