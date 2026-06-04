# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GPU acceptance coverage for Wan2.1-VACE native transformer features.

Uses the 1.3B variant for faster CI testing.

Coverage:
  Single GPU:
    - Cache-DiT + layerwise CPU offload
  Two GPUs:
    - Ulysses-SP = 2
    - Ring = 2
    - CFG-Parallel = 2
    - TP = 2 + VAE-Patch-Parallel = 2
    - HSDP = 2 + VAE-Patch-Parallel = 2
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
PROMPT = "A cat walking slowly across a sunlit garden path"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_vace_feature_cases():
    return [
        # Single GPU: Cache-DiT + layerwise CPU offload
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cache-backend",
                    "cache_dit",
                    "--enable-layerwise-offload",
                    "--vae-use-tiling",
                ],
            ),
            id="single_card_001",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
        # 2 GPUs: Ulysses-SP = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--usp",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_001",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: Ring = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--ring",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_002",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: CFG-Parallel = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--cfg-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_003",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: TP = 2 + VAE-Patch-Parallel = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--tensor-parallel-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_004",
            marks=PARALLEL_FEATURE_MARKS,
        ),
        # 2 GPUs: HSDP = 2 + VAE-Patch-Parallel = 2
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--use-hsdp",
                    "--hsdp-shard-size",
                    "2",
                    "--vae-patch-parallel-size",
                    "2",
                    "--vae-use-tiling",
                ],
            ),
            id="parallel_005",
            marks=PARALLEL_FEATURE_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_vace_feature_cases(),
    indirect=True,
)
def test_wan_2_1_vace(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    """Test VACE T2V generation with all supported diffusion acceleration features."""
    openai_client.send_video_diffusion_request(
        {
            "model": MODEL,
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
    )
