# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WAN21_MODEL_INDEX = PROJECT_ROOT / "vllm_omni" / "diffusion" / "models" / "wan2_1" / "model_index.py"
WAN21_COMMON = PROJECT_ROOT / "vllm_omni" / "diffusion" / "models" / "wan2_1" / "pipeline_wan2_1_common.py"
TEXT_TO_VIDEO_CURL = PROJECT_ROOT / "examples" / "online_serving" / "text_to_video" / "run_curl_text_to_video.sh"
TEXT_TO_VIDEO_SERVER = PROJECT_ROOT / "examples" / "online_serving" / "text_to_video" / "run_server.sh"
TEXT_TO_VIDEO_README = PROJECT_ROOT / "examples" / "online_serving" / "text_to_video" / "README.md"
IMAGE_TO_VIDEO_CURL = PROJECT_ROOT / "examples" / "online_serving" / "image_to_video" / "run_curl_image_to_video.sh"
VACE_EXAMPLE = PROJECT_ROOT / "examples" / "offline_inference" / "vace" / "vace_video_generation.py"
VACE_README = PROJECT_ROOT / "examples" / "offline_inference" / "vace" / "vace_video_generation.md"

_MODEL_INDEX_SPEC = importlib.util.spec_from_file_location(
    "wan21_model_index_under_test",
    WAN21_MODEL_INDEX,
)
assert _MODEL_INDEX_SPEC is not None
assert _MODEL_INDEX_SPEC.loader is not None
wan21_model_index = importlib.util.module_from_spec(_MODEL_INDEX_SPEC)
_MODEL_INDEX_SPEC.loader.exec_module(wan21_model_index)

WAN21_FLF2V_PIPELINE = getattr(wan21_model_index, "WAN21_FLF2V_PIPELINE", None)
WAN21_I2V_PIPELINE = wan21_model_index.WAN21_I2V_PIPELINE
WAN21_T2V_PIPELINE = wan21_model_index.WAN21_T2V_PIPELINE
WAN21_VACE_PIPELINE = wan21_model_index.WAN21_VACE_PIPELINE
resolve_wan21_pipeline_class_name = wan21_model_index.resolve_wan21_pipeline_class_name


OFFICIAL_T2V_IDS = (
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
)
OFFICIAL_I2V_IDS = (
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
)
OFFICIAL_FLF2V_ID = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
OFFICIAL_VACE_IDS = (
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "Wan-AI/Wan2.1-VACE-14B-diffusers",
)


def _load_common_flow_shift_helpers():
    tree = ast.parse(WAN21_COMMON.read_text(encoding="utf-8"))
    wanted = {
        "resolve_wan21_default_flow_shift",
        "resolve_wan21_flow_shift",
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(WAN21_COMMON), "exec"), namespace)
    return (
        namespace["resolve_wan21_default_flow_shift"],
        namespace["resolve_wan21_flow_shift"],
    )


def _class_method_def(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == method_name:
                return member
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


@pytest.mark.parametrize("model", OFFICIAL_T2V_IDS)
def test_official_wan21_t2v_ids_resolve_to_t2v_pipeline(model: str):
    assert resolve_wan21_pipeline_class_name(model, {"_class_name": "WanPipeline"}) == WAN21_T2V_PIPELINE


@pytest.mark.parametrize("model", OFFICIAL_I2V_IDS)
def test_official_wan21_i2v_ids_resolve_to_i2v_pipeline(model: str):
    assert (
        resolve_wan21_pipeline_class_name(
            model,
            {
                "_class_name": "WanImageToVideoPipeline",
                "image_encoder": ["transformers", "CLIPVisionModel"],
            },
        )
        == WAN21_I2V_PIPELINE
    )


def test_official_wan21_flf2v_id_resolves_before_generic_i2v():
    assert (
        resolve_wan21_pipeline_class_name(
            OFFICIAL_FLF2V_ID,
            {
                "_class_name": "WanImageToVideoPipeline",
                "image_encoder": ["transformers", "CLIPVisionModel"],
            },
        )
        == WAN21_FLF2V_PIPELINE
    )


@pytest.mark.parametrize("model", OFFICIAL_VACE_IDS)
def test_official_wan21_vace_ids_resolve_to_vace_pipeline(model: str):
    assert resolve_wan21_pipeline_class_name(model, {"_class_name": "WanVACEPipeline"}) == WAN21_VACE_PIPELINE


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (OFFICIAL_FLF2V_ID, 16.0),
        ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", 5.0),
        ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 3.0),
        ("Wan-AI/Wan2.1-T2V-14B-Diffusers", 3.0),
        ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", 3.0),
        ("Wan-AI/Wan2.1-VACE-1.3B-diffusers", 3.0),
        ("Wan-AI/Wan2.1-VACE-14B-diffusers", 3.0),
        (None, 3.0),
    ],
)
def test_wan21_default_flow_shift_is_model_aware(model: str | None, expected: float):
    resolve_wan21_default_flow_shift, _ = _load_common_flow_shift_helpers()

    assert resolve_wan21_default_flow_shift(model) == expected


def test_wan21_request_flow_shift_falls_back_to_model_aware_default():
    _, resolve_wan21_flow_shift = _load_common_flow_shift_helpers()
    req = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}))
    od_config = SimpleNamespace(
        flow_shift=None,
        model="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    )

    assert resolve_wan21_flow_shift(req, od_config) == 16.0


def test_wan21_request_flow_shift_keeps_config_and_request_overrides():
    _, resolve_wan21_flow_shift = _load_common_flow_shift_helpers()
    req = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={}))
    od_config = SimpleNamespace(flow_shift=7.0, model=OFFICIAL_FLF2V_ID)
    assert resolve_wan21_flow_shift(req, od_config) == 7.0

    req.sampling_params.extra_args["flow_shift"] = "9.5"
    assert resolve_wan21_flow_shift(req, od_config) == 9.5


def test_wan21_pipeline_constructor_uses_model_aware_default_flow_shift():
    init_fn = _class_method_def(WAN21_COMMON, "Wan21PipelineBase", "__init__")
    calls = [
        node
        for node in ast.walk(init_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_wan21_default_flow_shift"
    ]
    assert calls


def test_wan21_vace_pipeline_does_not_override_shared_flow_shift_default():
    source = WAN21_COMMON.read_text(encoding="utf-8")
    assert "replace(od_config, flow_shift=3.0)" not in source


def test_wan21_examples_do_not_force_flow_shift_when_env_unset():
    text_source = TEXT_TO_VIDEO_CURL.read_text(encoding="utf-8")
    image_source = IMAGE_TO_VIDEO_CURL.read_text(encoding="utf-8")
    vace_source = VACE_EXAMPLE.read_text(encoding="utf-8")

    for source in (text_source, image_source):
        assert 'if [ "${IS_WAN21}" = "1" ]; then' in source
        assert 'if [ -z "${FLOW_SHIFT:-}" ]; then' in source
        assert 'if [ -z "${FLOW_SHIFT+x}" ]; then' not in source
        assert 'FLOW_SHIFT="16.0"' in source
        assert 'FLOW_SHIFT="5.0"' in source
        assert 'FLOW_SHIFT="3.0"' in source
    assert 'FLOW_SHIFT="${FLOW_SHIFT:-5.0}"' not in image_source
    assert 'FLOW_SHIFT="${FLOW_SHIFT:-12.0}"' in image_source
    assert '"--flow-shift", type=float, default=None' in vace_source


def test_wan21_run_server_does_not_force_flow_shift_when_env_unset():
    source = TEXT_TO_VIDEO_SERVER.read_text(encoding="utf-8")

    assert 'FLOW_SHIFT="${FLOW_SHIFT:-5.0}"' not in source
    assert 'FLOW_SHIFT="${FLOW_SHIFT:-}"' in source
    assert 'FLOW_SHIFT_FLAG=""' in source
    assert '--flow-shift "$FLOW_SHIFT"' not in source
    assert "$FLOW_SHIFT_FLAG" in source
    assert 'elif [[ "$MODEL" == *"Wan2.2"* ]]; then' in source
    assert 'FLOW_SHIFT_FLAG="--flow-shift 5.0"' in source


def test_wan21_docs_describe_model_aware_flow_shift_defaults():
    text_to_video_readme = TEXT_TO_VIDEO_README.read_text(encoding="utf-8")
    vace_readme = VACE_README.read_text(encoding="utf-8")

    assert "Wan2.1 uses model-aware flow-shift defaults" in text_to_video_readme
    assert "`3.0` for T2V/I2V-480P/VACE" in text_to_video_readme
    assert "`5.0` for I2V-720P" in text_to_video_readme
    assert "`16.0` for FLF2V-720P" in text_to_video_readme
    assert "| `--flow-shift` | float | `None` | Scheduler flow shift parameter" in vace_readme
