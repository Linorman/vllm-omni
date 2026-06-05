# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GPU acceptance coverage for Wan2.1 T2V/I2V/FLF2V native transformer features.

Coverage:
  Single GPU:
    - Cache-DiT + layerwise CPU offload
  Two GPUs:
    - Ulysses-SP
    - Ring
    - CFG-Parallel
    - Pipeline-Parallel
    - Tensor-Parallel + VAE-Patch-Parallel
    - HSDP + VAE-Patch-Parallel
"""

from pathlib import Path

import pytest

from tests.e2e.accuracy.wan21.video_metadata import ffprobe_video
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

PROMPT = "A small robot carefully watering bright flowers in a quiet greenhouse."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text, distorted"

WAN21_MODELS = [
    ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "t2v_13b", None),
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "t2v_14b", None),
    ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "i2v_480p", "image"),
    ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", "i2v_720p", "image"),
    ("Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers", "flf2v_720p", "two_images"),
]

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)

FEATURE_CONFIGS = [
    (
        "cache_dit_layerwise_offload",
        ["--cache-backend", "cache_dit", "--enable-layerwise-offload"],
        SINGLE_CARD_MARKS,
    ),
    ("ulysses_sp", ["--usp", "2"], PARALLEL_MARKS),
    ("ring", ["--ring", "2"], PARALLEL_MARKS),
    ("cfg_parallel", ["--cfg-parallel-size", "2"], PARALLEL_MARKS),
    (
        "tensor_parallel_vae_patch",
        [
            "--tensor-parallel-size",
            "2",
            "--vae-patch-parallel-size",
            "2",
            "--vae-use-tiling",
        ],
        PARALLEL_MARKS,
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
        PARALLEL_MARKS,
    ),
    ("pipeline_parallel", ["--pipeline-parallel-size", "2"], PARALLEL_MARKS),
]


def _get_wan21_feature_cases():
    cases = []
    for model_path, model_key, _input_mode in WAN21_MODELS:
        for feature_id, server_args, marks in FEATURE_CONFIGS:
            cases.append(
                pytest.param(
                    OmniServerParams(model=model_path, server_args=server_args),
                    id=f"{model_key}_{feature_id}",
                    marks=marks,
                )
            )
    return cases


@pytest.mark.parametrize(
    "omni_server",
    _get_wan21_feature_cases(),
    indirect=True,
)
def test_wan_2_1_t2v_i2v_flf2v_features(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    tmp_path: Path,
):
    model_path = omni_server.model
    input_mode = next(mode for model, _key, mode in WAN21_MODELS if model == model_path)
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

    if input_mode == "image":
        request_config["image_reference"] = (
            f"data:image/jpeg;base64,{generate_synthetic_image(320, 480)['base64']}"
        )
    elif input_mode == "two_images":
        request_config["image_reference"] = (
            f"data:image/jpeg;base64,{generate_synthetic_image(320, 480)['base64']}"
        )
        request_config["last_image_reference"] = (
            "data:image/jpeg;base64,"
            f"{generate_synthetic_image(320, 480, color=(64, 128, 255))['base64']}"
        )

    result = openai_client.send_video_diffusion_request(request_config)[0]
    assert result.videos is not None
    assert len(result.videos) == 1
    assert len(result.videos[0]) > 1024

    video_path = tmp_path / "wan21_output.mp4"
    video_path.write_bytes(result.videos[0])
    metadata = ffprobe_video(video_path)
    assert int(metadata["width"]) > 0
    assert int(metadata["height"]) > 0
