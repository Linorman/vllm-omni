# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GPU acceptance coverage for Wan2.1 T2V/I2V native transformer features.

Coverage:
  Single GPU:
    - Cache-DiT + layerwise CPU offload
  Two GPUs:
    - Ulysses-SP
    - Ring
    - CFG-Parallel
    - Tensor-Parallel + VAE-Patch-Parallel
    - HSDP + VAE-Patch-Parallel
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

PROMPT = "A small robot carefully watering bright flowers in a quiet greenhouse."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text, distorted"

WAN21_MODELS = [
    ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "t2v"),
    ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "i2v"),
]

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

CACHE_DIT_OFFLOAD_ARGS = [
    "--cache-backend",
    "cache_dit",
    "--enable-layerwise-offload",
]

PARALLEL_CONFIGS = [
    ("ulysses_sp", ["--usp", "2"]),
    ("ring", ["--ring", "2"]),
    ("cfg_parallel", ["--cfg-parallel-size", "2"]),
    (
        "tp_vae_patch",
        [
            "--tensor-parallel-size",
            "2",
            "--vae-patch-parallel-size",
            "2",
            "--vae-use-tiling",
        ],
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
    ),
]


def _get_wan21_feature_cases():
    cases = []
    for model_path, model_key in WAN21_MODELS:
        cases.append(
            pytest.param(
                OmniServerParams(model=model_path, server_args=CACHE_DIT_OFFLOAD_ARGS),
                id=f"{model_key}_cache_dit_layerwise_offload",
                marks=SINGLE_CARD_MARKS,
            )
        )
        for feature_id, server_args in PARALLEL_CONFIGS:
            cases.append(
                pytest.param(
                    OmniServerParams(model=model_path, server_args=server_args),
                    id=f"{model_key}_{feature_id}",
                    marks=PARALLEL_MARKS,
                )
            )
    return cases


@pytest.mark.parametrize(
    "omni_server",
    _get_wan21_feature_cases(),
    indirect=True,
)
def test_wan_2_1_t2v_i2v_features(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    model_path = omni_server.model
    request_config = {
        "model": model_path,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": 480,
            "width": 320,
            "num_frames": 5,
            "fps": 8,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }

    if "I2V" in model_path:
        request_config["image_reference"] = (
            f"data:image/jpeg;base64,{generate_synthetic_image(320, 480)['base64']}"
        )

    openai_client.send_video_diffusion_request(request_config)
