# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion import diffusion_engine
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_dummy_run_num_frames_uses_explicit_model_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    class JointAudioVideoModel:
        dummy_run_num_frames = 2

    monkeypatch.setattr(
        diffusion_engine.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: JointAudioVideoModel,
    )

    assert diffusion_engine.get_dummy_run_num_frames("joint_audio_video", supports_audio_input=False) == 2


def test_dummy_run_num_frames_keeps_audio_output_default(monkeypatch: pytest.MonkeyPatch) -> None:
    class AudioOutputModel:
        support_audio_output = True

    monkeypatch.setattr(
        diffusion_engine.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: AudioOutputModel,
    )

    assert diffusion_engine.get_dummy_run_num_frames("audio_output", supports_audio_input=False) == 2


def test_dummy_run_num_frames_defaults_to_single_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    class VideoOnlyModel:
        pass

    monkeypatch.setattr(
        diffusion_engine.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: VideoOnlyModel,
    )

    assert diffusion_engine.get_dummy_run_num_frames("video_only", supports_audio_input=False) == 1


def test_dummy_run_num_frames_uses_audio_input_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diffusion_engine.DiffusionModelRegistry,
        "_try_load_model_cls",
        lambda model_class_name: None,
    )

    assert diffusion_engine.get_dummy_run_num_frames("unknown", supports_audio_input=True) == 2


def test_dummy_run_supplies_two_images_for_two_image_models(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(model_class_name="Wan21FLF2VPipeline")

    monkeypatch.setattr(diffusion_engine, "supports_multimodal_input", lambda od_config: (True, False))
    monkeypatch.setattr(diffusion_engine, "image_color_format", lambda model_class_name: "RGB")

    captured = {}

    def _capture_pre_process(request):
        captured["request"] = request
        return request

    engine.pre_process_func = _capture_pre_process
    engine.add_req_and_wait_for_response = lambda request: SimpleNamespace(error=None)

    engine._dummy_run()

    image_input = captured["request"].prompts[0]["multi_modal_data"]["image"]
    assert isinstance(image_input, list)
    assert len(image_input) == 2
    assert image_input[0].size == (512, 512)
    assert image_input[1].size == (512, 512)
    assert captured["request"].sampling_params.num_frames == 5
