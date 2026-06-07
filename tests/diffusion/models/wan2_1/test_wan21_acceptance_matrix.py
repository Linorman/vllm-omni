# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ACCURACY_COMMON = PROJECT_ROOT / "tests" / "e2e" / "accuracy" / "wan21" / "wan21_video_similarity_common.py"
ACCURACY_TEST = PROJECT_ROOT / "tests" / "e2e" / "accuracy" / "wan21" / "test_wan21_video_similarity.py"
BUILDKITE_NIGHTLY = PROJECT_ROOT / ".buildkite" / "test-nightly.yml"
MODEL_ASSETS = PROJECT_ROOT / "tests" / "e2e" / "accuracy" / "wan21" / "MODEL_ASSETS.md"
SUPPORTED_MODELS = PROJECT_ROOT / "docs" / "models" / "supported_models.md"
CI_WAN21_ASSET_DIR = "/tmp/wan21-test-assets"
CI_WAN21_IMAGE_SOURCE = f"{CI_WAN21_ASSET_DIR}/input.png"
CI_WAN21_LAST_IMAGE_SOURCE = f"{CI_WAN21_ASSET_DIR}/last.png"

OFFICIAL_WAN21_MODEL_IDS = (
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "Wan-AI/Wan2.1-VACE-14B-diffusers",
)

ACCURACY_CASE_MODEL_PAIRS = (
    ("MODEL_T2V_13B", "t2v_13b"),
    ("MODEL_T2V_14B", "t2v_14b"),
    ("MODEL_I2V_480P", "i2v_480p"),
    ("MODEL_I2V_720P", "i2v_720p"),
    ("MODEL_FLF2V_720P", "flf2v_720p"),
    ("MODEL_VACE_13B", "vace_13b_reference_image"),
    ("MODEL_VACE_14B", "vace_14b_reference_image"),
)

RELEASE_GATE_MODEL_NAMES = tuple(model_constant for model_constant, _ in ACCURACY_CASE_MODEL_PAIRS)

NIGHTLY_SMOKE_CASE_IDS = (
    "t2v_13b_cache_dit_layerwise_offload",
    "t2v_14b_ulysses_sp",
    "i2v_480p_ring",
    "i2v_720p_tensor_parallel_vae_patch",
    "flf2v_720p_pipeline_parallel",
    "vace_13b_reference_image",
    "vace_14b_cfg_parallel",
)


def _parse_python(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignment_value(module: ast.Module, name: str) -> ast.expr | None:
    for node in module.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return node.value

    return None


def _string_constants(module: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}

    for node in module.body:
        if isinstance(node, ast.Assign):
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            if isinstance(node.target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                constants[node.target.id] = value.value

    return constants


def _name_tuple(module: ast.Module, name: str) -> set[str]:
    value = _assignment_value(module, name)
    if value is None:
        return set()

    assert isinstance(value, ast.Tuple), f"{name} must be a tuple literal"

    names: set[str] = set()
    for element in value.elts:
        assert isinstance(element, ast.Name), f"{name} must contain only constant names"
        names.add(element.id)

    return names


def _wan21_case_model_pairs(module: ast.Module) -> set[tuple[str, str]]:
    value = _assignment_value(module, "WAN21_CASE_SPECS")
    if value is None:
        return set()

    assert isinstance(value, (ast.List, ast.Tuple)), "WAN21_CASE_SPECS must be a list/tuple literal"

    pairs: set[tuple[str, str]] = set()
    for element in value.elts:
        assert isinstance(element, ast.Tuple), "WAN21_CASE_SPECS entries must be tuple literals"
        assert len(element.elts) >= 2, "WAN21_CASE_SPECS entries must include model and case id"

        model_constant, case_id = element.elts[:2]
        assert isinstance(model_constant, ast.Name), "WAN21_CASE_SPECS model must be a constant name"
        assert isinstance(case_id, ast.Constant) and isinstance(case_id.value, str), (
            "WAN21_CASE_SPECS case id must be a string literal"
        )
        pairs.add((model_constant.id, case_id.value))

    return pairs


def _wan21_nightly_smoke_command(source: str) -> str:
    commands = [
        line.strip()
        for line in source.splitlines()
        if "python -m pytest" in line
        and "tests/e2e/online_serving/test_wan_2_1_expansion.py" in line
        and "tests/e2e/online_serving/test_wan_2_1_vace_expansion.py" in line
    ]

    assert commands, "missing Wan2.1 nightly smoke command for expansion and VACE tests"
    assert len(commands) == 1, f"expected one Wan2.1 nightly smoke command, found {commands!r}"
    return commands[0]


def _wan21_accuracy_command_index(source: str) -> tuple[int, str]:
    commands = [
        (index, line.strip())
        for index, line in enumerate(source.splitlines())
        if "python -m pytest" in line and "tests/e2e/accuracy/wan21/test_wan21_video_similarity.py" in line
    ]

    assert commands, "missing Wan2.1 accuracy command"
    assert len(commands) == 1, f"expected one Wan2.1 accuracy command, found {commands!r}"
    return commands[0]


def _pytest_k_selector(command: str) -> str:
    marker = ' -k "'
    assert marker in command, f"Wan2.1 nightly smoke command is missing pytest -k selector: {command!r}"
    return command.split(marker, 1)[1].split('"', 1)[0]


def _or_selector_terms(selector: str) -> set[str]:
    return {term.strip("() ") for term in selector.split(" or ") if term.strip("() ")}


def _assert_exact_set(actual, expected, label: str) -> None:
    missing = expected - actual
    extra = actual - expected
    assert actual == expected, f"{label} mismatch; missing={sorted(missing)!r}; extra={sorted(extra)!r}"


def test_wan21_release_gate_includes_all_official_diffusers_models():
    module = _parse_python(ACCURACY_COMMON)
    model_ids = {value for name, value in _string_constants(module).items() if name.startswith("MODEL_")}
    release_gate_names = _name_tuple(module, "RELEASE_GATE_MODELS")

    _assert_exact_set(
        model_ids,
        set(OFFICIAL_WAN21_MODEL_IDS),
        "Wan2.1 model id constants",
    )
    _assert_exact_set(
        release_gate_names,
        set(RELEASE_GATE_MODEL_NAMES),
        "RELEASE_GATE_MODELS constants",
    )


def test_wan21_similarity_cases_cover_release_gate_models():
    module = _parse_python(ACCURACY_TEST)

    _assert_exact_set(
        _wan21_case_model_pairs(module),
        set(ACCURACY_CASE_MODEL_PAIRS),
        "WAN21_CASE_SPECS case/model pairs",
    )


def test_wan21_nightly_smoke_covers_architectures_and_large_variants():
    source = BUILDKITE_NIGHTLY.read_text(encoding="utf-8")
    command = _wan21_nightly_smoke_command(source)
    selector = _pytest_k_selector(command)

    _assert_exact_set(
        _or_selector_terms(selector),
        set(NIGHTLY_SMOKE_CASE_IDS),
        "Wan2.1 nightly smoke case ids",
    )


def test_wan21_nightly_accuracy_prepares_default_image_assets():
    source = BUILDKITE_NIGHTLY.read_text(encoding="utf-8")
    lines = source.splitlines()
    accuracy_index, accuracy_command = _wan21_accuracy_command_index(source)
    preceding_command = next(
        (line.strip() for line in reversed(lines[:accuracy_index]) if line.strip().startswith("- ")),
        "",
    )

    assert f'WAN21_IMAGE_SOURCE="${{WAN21_IMAGE_SOURCE:-{CI_WAN21_IMAGE_SOURCE}}}"' in accuracy_command
    assert f'WAN21_LAST_IMAGE_SOURCE="${{WAN21_LAST_IMAGE_SOURCE:-{CI_WAN21_LAST_IMAGE_SOURCE}}}"' in accuracy_command
    assert preceding_command.startswith("- python -c ")
    assert f'Path("{CI_WAN21_ASSET_DIR}")' in preceding_command
    assert "out.mkdir(parents=True, exist_ok=True)" in preceding_command
    assert '.save(out / "input.png")' in preceding_command
    assert '.save(out / "last.png")' in preceding_command
    assert "from PIL import Image" in preceding_command
    assert 'Image.new("RGB"' in preceding_command


def test_wan21_model_assets_document_release_gate_models():
    source = MODEL_ASSETS.read_text(encoding="utf-8")

    assert "Release-gated GPU validation set" in source
    for model_id in OFFICIAL_WAN21_MODEL_IDS:
        assert model_id in source


def test_wan21_supported_models_document_scope_limits():
    source = SUPPORTED_MODELS.read_text(encoding="utf-8")

    assert "The Wan2.1 entries above cover the official Diffusers video/VACE checkpoint" in source
    assert "They do not include Wan2.1 S2V or a Wan2.1" in source
    assert "image-generation endpoint" in source
