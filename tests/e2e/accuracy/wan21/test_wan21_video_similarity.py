from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from base64 import b64decode, b64encode
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from PIL import Image

from tests.e2e.accuracy.helpers import (
    assert_video_metadata,
    assert_video_similarity_metrics,
    build_online_image_reference,
    probe_binary,
    probe_video,
    validate_image_source,
)
from tests.e2e.accuracy.wan21.wan21_video_similarity_common import (
    CONDITIONING_SCALE,
    FLOW_SHIFT_BY_MODEL,
    FPS,
    GUIDANCE_SCALE,
    HEIGHT,
    MODEL_FLF2V_720P,
    MODEL_I2V_480P,
    MODEL_I2V_720P,
    MODEL_T2V_13B,
    MODEL_T2V_14B,
    MODEL_VACE_13B,
    MODEL_VACE_14B,
    NEGATIVE_PROMPT,
    NUM_FRAMES,
    NUM_INFERENCE_STEPS,
    PROMPT_BY_MODEL,
    PSNR_THRESHOLD,
    SEED,
    SIZE,
    SSIM_THRESHOLD,
    WIDTH,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = REPO_ROOT.parent
RUNNER_PATH = Path(__file__).with_name("run_wan21_diffusers_reference.py")
RESULT_ROOT = Path(__file__).parent / "result"
VIDEO_TIMEOUT_SECONDS = 60 * 60

WAN21_CASE_SPECS = [
    (MODEL_T2V_13B, "t2v_13b", None),
    (MODEL_T2V_14B, "t2v_14b", None),
    (MODEL_I2V_480P, "i2v_480p", "image"),
    (MODEL_I2V_720P, "i2v_720p", "image"),
    (MODEL_FLF2V_720P, "flf2v_720p", "two_images"),
    (MODEL_VACE_13B, "vace_13b_reference_image", "reference_image"),
    (MODEL_VACE_14B, "vace_14b_reference_image", "reference_image"),
]

WAN21_CASES = [
    pytest.param(model, case_key, input_mode, id=case_key)
    for model, case_key, input_mode in WAN21_CASE_SPECS
]

SERVER_CASES = [
    pytest.param(
        OmniServerParams(model=model, server_args=["--enforce-eager"], use_omni=True),
        case_key,
        input_mode,
        id=f"{case_key}_online",
    )
    for model, case_key, input_mode in WAN21_CASE_SPECS
]


@pytest.fixture
def wan21_image_source() -> str | None:
    return os.environ.get("WAN21_IMAGE_SOURCE")


@pytest.fixture
def wan21_last_image_source() -> str | None:
    return os.environ.get("WAN21_LAST_IMAGE_SOURCE")


def _runner_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT), str(WORKSPACE_ROOT / "diffusers" / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def _artifact_paths(case_key: str) -> tuple[Path, Path, Path]:
    safe_key = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in case_key)
    digest = sha1(case_key.encode("utf-8")).hexdigest()[:8]
    artifact_dir = RESULT_ROOT / f"{safe_key}-{digest}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / "online.mp4", artifact_dir / "offline.mp4", artifact_dir / "offline_metadata.json"


def _build_diffusers_command(
    *,
    model: str,
    output_path: Path,
    metadata_path: Path,
    image: str | None,
    last_image: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--model",
        model,
        "--prompt",
        PROMPT_BY_MODEL[model],
        "--negative-prompt",
        NEGATIVE_PROMPT,
        "--output",
        str(output_path),
        "--metadata-output",
        str(metadata_path),
        "--height",
        str(HEIGHT),
        "--width",
        str(WIDTH),
        "--num-frames",
        str(NUM_FRAMES),
        "--num-inference-steps",
        str(NUM_INFERENCE_STEPS),
        "--guidance-scale",
        str(GUIDANCE_SCALE),
        "--flow-shift",
        str(FLOW_SHIFT_BY_MODEL[model]),
        "--seed",
        str(SEED),
        "--fps",
        str(FPS),
    ]
    if image is not None:
        command.extend(["--image", image])
    if last_image is not None:
        command.extend(["--last-image", last_image])
    if "VACE" in model:
        command.extend(["--conditioning-scale", str(CONDITIONING_SCALE)])
    return command


def _add_reference_file(
    *,
    files: dict[str, tuple[str, BytesIO, str]],
    field_name: str,
    source: str,
    filename_prefix: str,
) -> str | None:
    if not source.startswith("data:image"):
        return source
    header, encoded = source.split(",", 1)
    content_type = header.split(";")[0].removeprefix("data:")
    extension = content_type.split("/")[-1]
    files[field_name] = (
        f"{filename_prefix}.{extension}",
        BytesIO(b64decode(encoded)),
        content_type,
    )
    return None


def _send_video_request_with_references(
    openai_client,
    request_config: dict[str, Any],
    *,
    timeout_seconds: int,
) -> bytes:
    form_data = request_config.get("form_data")
    if not isinstance(form_data, dict):
        raise ValueError("Video request_config must contain 'form_data'")
    normalized_form_data = {key: str(value) for key, value in form_data.items() if value is not None}
    files: dict[str, tuple[str, BytesIO, str]] = {}

    image_reference = request_config.get("image_reference")
    if image_reference:
        remote = _add_reference_file(
            files=files,
            field_name="input_reference",
            source=image_reference,
            filename_prefix="reference",
        )
        if remote is not None:
            normalized_form_data["image_reference"] = json.dumps({"image_url": remote})

    last_image_reference = request_config.get("last_image_reference")
    if last_image_reference:
        remote = _add_reference_file(
            files=files,
            field_name="last_input_reference",
            source=last_image_reference,
            filename_prefix="last_reference",
        )
        if remote is not None:
            normalized_form_data["last_image_reference"] = json.dumps({"image_url": remote})

    start_time = time.perf_counter()
    response = requests.post(
        openai_client._build_url("/v1/videos"),
        data=normalized_form_data,
        files=files,
        headers={"Accept": "application/json"},
        timeout=60,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc}; response body: {response.text}", response=response) from exc
    video_id = response.json()["id"]
    openai_client._wait_until_video_completed(video_id, timeout_seconds=timeout_seconds)
    video_content = openai_client._download_video_content(video_id)
    print(f"online_video_e2e_latency_s={time.perf_counter() - start_time:.3f}")
    return video_content


def _request_config(*, model: str, image: str | None, last_image: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": model,
        "form_data": {
            "prompt": PROMPT_BY_MODEL[model],
            "negative_prompt": NEGATIVE_PROMPT,
            "size": SIZE,
            "fps": FPS,
            "num_frames": NUM_FRAMES,
            "guidance_scale": GUIDANCE_SCALE,
            "flow_shift": FLOW_SHIFT_BY_MODEL[model],
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
        },
    }
    if image is not None:
        config["image_reference"] = build_online_image_reference(image)
    if last_image is not None:
        config["last_image_reference"] = build_online_image_reference(last_image)
    return config


def _resolve_assets(
    *,
    input_mode: str | None,
    image_source: str | None,
    last_image_source: str | None,
) -> tuple[str | None, str | None]:
    image = image_source if input_mode in {"image", "two_images", "reference_image"} else None
    last_image = last_image_source if input_mode == "two_images" else None
    return image, last_image


def _validate_required_assets(*, input_mode: str | None, image: str | None, last_image: str | None) -> None:
    if input_mode in {"image", "two_images", "reference_image"} and image is None:
        pytest.skip("WAN21_IMAGE_SOURCE is required for this Wan2.1 accuracy case.")
    if input_mode == "two_images" and last_image is None:
        pytest.skip("WAN21_LAST_IMAGE_SOURCE is required for this Wan2.1 FLF2V accuracy case.")
    if image is not None:
        validate_image_source(image)
    if last_image is not None:
        validate_image_source(last_image)


def test_build_diffusers_command_includes_metadata_output(tmp_path: Path) -> None:
    command = _build_diffusers_command(
        model=MODEL_I2V_480P,
        output_path=tmp_path / "offline.mp4",
        metadata_path=tmp_path / "offline_metadata.json",
        image="input.png",
        last_image=None,
    )

    assert command[:2] == [sys.executable, str(RUNNER_PATH)]
    assert "--metadata-output" in command
    assert "--image" in command
    assert "--flow-shift" in command


def test_request_config_encodes_first_and_last_image_references(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    Image.new("RGB", (4, 2), color=(10, 20, 30)).save(first)
    Image.new("RGB", (4, 2), color=(30, 20, 10)).save(last)

    config = _request_config(model=MODEL_FLF2V_720P, image=str(first), last_image=str(last))

    assert config["image_reference"].startswith("data:image/png;base64,")
    assert config["last_image_reference"].startswith("data:image/png;base64,")


def test_add_reference_file_returns_remote_image_url_for_non_data_reference() -> None:
    files: dict[str, tuple[str, BytesIO, str]] = {}

    remote = _add_reference_file(
        files=files,
        field_name="input_reference",
        source="https://example.test/input.png",
        filename_prefix="reference",
    )

    assert remote == "https://example.test/input.png"
    assert files == {}


def test_add_reference_file_materializes_data_url() -> None:
    image = Image.new("RGB", (2, 2), color=(1, 2, 3))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    source = f"data:image/png;base64,{b64encode(buffer.getvalue()).decode('ascii')}"
    files: dict[str, tuple[str, BytesIO, str]] = {}

    remote = _add_reference_file(
        files=files,
        field_name="input_reference",
        source=source,
        filename_prefix="reference",
    )

    assert remote is None
    filename, payload, content_type = files["input_reference"]
    assert filename == "reference.png"
    assert content_type == "image/png"
    assert len(payload.getvalue()) > 0


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("model,case_key,input_mode", WAN21_CASES)
def test_wan21_diffusers_offline_generates_video(
    model: str,
    case_key: str,
    input_mode: str | None,
    wan21_image_source: str | None,
    wan21_last_image_source: str | None,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Wan2.1 Diffusers offline accuracy test requires CUDA.")
    probe_binary("ffprobe")
    if not RUNNER_PATH.exists():
        raise AssertionError(f"Offline diffusers runner does not exist: {RUNNER_PATH}")

    image, last_image = _resolve_assets(
        input_mode=input_mode,
        image_source=wan21_image_source,
        last_image_source=wan21_last_image_source,
    )
    _validate_required_assets(input_mode=input_mode, image=image, last_image=last_image)
    _, offline_path, metadata_path = _artifact_paths(case_key)
    subprocess.run(
        _build_diffusers_command(
            model=model,
            output_path=offline_path,
            metadata_path=metadata_path,
            image=image,
            last_image=last_image,
        ),
        cwd=REPO_ROOT,
        env=_runner_env(),
        check=True,
        timeout=VIDEO_TIMEOUT_SECONDS,
    )
    assert offline_path.exists(), f"Expected offline video artifact at {offline_path}"
    assert metadata_path.exists(), f"Expected offline metadata artifact at {metadata_path}"
    assert_video_metadata(probe_video(offline_path), width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server,case_key,input_mode", SERVER_CASES, indirect=["omni_server"])
def test_wan21_online_serving_generates_video(
    omni_server,
    openai_client,
    case_key: str,
    input_mode: str | None,
    wan21_image_source: str | None,
    wan21_last_image_source: str | None,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Wan2.1 online accuracy test requires CUDA.")
    probe_binary("ffprobe")

    image, last_image = _resolve_assets(
        input_mode=input_mode,
        image_source=wan21_image_source,
        last_image_source=wan21_last_image_source,
    )
    _validate_required_assets(input_mode=input_mode, image=image, last_image=last_image)
    online_path, _, _ = _artifact_paths(case_key)
    video_bytes = _send_video_request_with_references(
        openai_client,
        _request_config(model=omni_server.model, image=image, last_image=last_image),
        timeout_seconds=VIDEO_TIMEOUT_SECONDS,
    )
    online_path.write_bytes(video_bytes)
    assert online_path.exists(), f"Expected online video artifact at {online_path}"
    assert_video_metadata(probe_video(online_path), width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)


@pytest.mark.benchmark
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("model,case_key,input_mode", WAN21_CASES)
def test_wan21_serving_matches_diffusers_video_similarity(
    model: str,
    case_key: str,
    input_mode: str | None,
) -> None:
    del model, input_mode
    if not torch.cuda.is_available():
        pytest.skip("Wan2.1 similarity e2e test requires CUDA.")
    probe_binary("ffmpeg")
    probe_binary("ffprobe")
    online_path, offline_path, metadata_path = _artifact_paths(case_key)
    if not online_path.exists():
        pytest.skip(f"Missing online artifact from prerequisite test: {online_path}")
    if not offline_path.exists() or not metadata_path.exists():
        pytest.skip(f"Missing offline artifacts from prerequisite test: {offline_path}, {metadata_path}")

    online_metadata = probe_video(online_path)
    offline_metadata = probe_video(offline_path)
    assert online_metadata == offline_metadata, (
        f"Video metadata mismatch:\n"
        f"online={online_metadata}\n"
        f"offline={offline_metadata}\n"
        f"online_path={online_path}\n"
        f"offline_path={offline_path}"
    )
    assert_video_metadata(online_metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)
    assert_video_similarity_metrics(
        label=f"wan21_{case_key}",
        online_path=online_path,
        offline_path=offline_path,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )
    print(f"offline_metadata={metadata_path}")
