# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WAN21_DIR = PROJECT_ROOT / "vllm_omni" / "diffusion" / "models" / "wan2_1"
ATTENTION_LAYER = PROJECT_ROOT / "vllm_omni" / "diffusion" / "attention" / "layer.py"
ATTENTION_REGISTRY = PROJECT_ROOT / "vllm_omni" / "diffusion" / "attention" / "backends" / "registry.py"
ATTENTION_SDPA = PROJECT_ROOT / "vllm_omni" / "diffusion" / "attention" / "backends" / "sdpa.py"
CACHE_BACKEND = PROJECT_ROOT / "vllm_omni" / "diffusion" / "cache" / "cache_dit_backend.py"
REGISTRY = PROJECT_ROOT / "vllm_omni" / "diffusion" / "registry.py"
WAN21_COMMON = WAN21_DIR / "pipeline_wan2_1_common.py"
WAN21_TRANSFORMER = WAN21_DIR / "wan2_1_transformer.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(_source(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _assigned_literal(path: Path, name: str):
    tree = ast.parse(_source(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def _assigned_dict(path: Path, name: str):
    value = _assigned_literal(path, name)
    assert isinstance(value, dict)
    return value


def _module_attrs(path: Path) -> set[str]:
    tree = ast.parse(_source(path))
    attrs = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            attrs.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            attrs.update(alias.asname or alias.name for alias in node.names)
    return attrs


def _function_def(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(_source(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _class_method_def(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(_source(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == method_name:
                return member
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _raises_value_error_with(node: ast.Raise, text: str) -> bool:
    call = node.exc
    if not isinstance(call, ast.Call):
        return False
    if not isinstance(call.func, ast.Name) or call.func.id != "ValueError":
        return False
    if not call.args or not isinstance(call.args[0], ast.Constant):
        return False
    return call.args[0].value == text


def _is_name_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _make_layers_factory_lambdas(path: Path) -> list[ast.Lambda]:
    tree = ast.parse(_source(path))
    lambdas: list[ast.Lambda] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "make_layers":
            continue
        assert len(node.args) >= 2
        factory = node.args[1]
        assert isinstance(factory, ast.Lambda), path
        lambdas.append(factory)
    return lambdas


def _is_set_current_diffusion_config_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set_current_diffusion_config"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "od_config"
    )


def _contains_transformer_creation(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        assigns_self_transformer = any(
            isinstance(target, ast.Attribute)
            and target.attr == "transformer"
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            for target in child.targets
        )
        if not assigns_self_transformer:
            continue
        value = child.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "_create_transformer"
        ):
            return True
    return False


def test_wan21_model_sources_do_not_import_wan22_modules():
    for path in WAN21_DIR.glob("*.py"):
        assert not any("wan2_2" in module for module in _imports_from(path)), path


def test_wan21_model_sources_do_not_wrap_diffusers_pipelines():
    forbidden_pipeline_classes = {
        "WanPipeline",
        "WanImageToVideoPipeline",
        "WanVACEPipeline",
    }
    for path in WAN21_DIR.glob("*.py"):
        tree = ast.parse(_source(path))
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert imported_names.isdisjoint(forbidden_pipeline_classes), path


def test_wan21_pipeline_uses_native_transformer_modules():
    source = _source(WAN21_COMMON)
    assert "from diffusers import WanTransformer3DModel" not in source
    assert "from diffusers import WanVACETransformer3DModel" not in source
    assert "from .wan2_1_transformer import Wan21Transformer3DModel" in source
    assert "from .wan2_1_vace_transformer import Wan21VACETransformer3DModel" in source


def test_wan21_entrypoints_are_wan21_native_wrappers():
    expected_bases = {
        "pipeline_wan2_1.py": ("Wan21Pipeline", "Wan21PipelineBase"),
        "pipeline_wan2_1_i2v.py": ("Wan21I2VPipeline", "Wan21I2VPipelineBase"),
        "pipeline_wan2_1_vace.py": ("Wan21VACEPipeline", "Wan21VACEPipelineBase"),
    }
    for filename, (class_name, base_name) in expected_bases.items():
        tree = ast.parse(_source(WAN21_DIR / filename))
        classes = {
            node.name: [base.id for base in node.bases if isinstance(base, ast.Name)]
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        assert classes[class_name] == [base_name]


def test_wan21_cache_dit_uses_independent_wan21_enabler():
    source = _source(CACHE_BACKEND)
    assert "def enable_cache_for_wan21(" in source
    assert '"Wan21Pipeline": enable_cache_for_wan21' in source
    assert '"Wan21I2VPipeline": enable_cache_for_wan21' in source
    assert '"Wan21VACEPipeline": enable_cache_for_wan21' in source
    assert '"Wan21Pipeline": enable_cache_for_wan22' not in source
    assert '"Wan21I2VPipeline": enable_cache_for_wan22' not in source
    assert '"Wan21VACEPipeline": enable_cache_for_wan22' not in source


def test_wan21_registered_process_functions_resolve():
    models = _assigned_dict(REGISTRY, "_DIFFUSION_MODELS")
    pre_process_funcs = _assigned_dict(REGISTRY, "_DIFFUSION_PRE_PROCESS_FUNCS")
    post_process_funcs = _assigned_dict(REGISTRY, "_DIFFUSION_POST_PROCESS_FUNCS")
    for arch in ("Wan21Pipeline", "Wan21I2VPipeline", "Wan21VACEPipeline"):
        _, mod_relname, _ = models[arch]
        module_attrs = _module_attrs(WAN21_DIR / f"{mod_relname}.py")
        process_func_names = {
            pre_process_funcs[arch],
            post_process_funcs[arch],
        }

        for func_name in process_func_names:
            assert func_name in module_attrs, (arch, func_name)


def test_wan21_package_exports_match_registered_entrypoints():
    package_exports = set(_assigned_literal(WAN21_DIR / "__init__.py", "__all__"))
    expected_attrs = {
        "Wan21Pipeline",
        "get_wan21_post_process_func",
        "get_wan21_pre_process_func",
        "Wan21I2VPipeline",
        "get_wan21_i2v_post_process_func",
        "get_wan21_i2v_pre_process_func",
        "Wan21VACEPipeline",
        "get_wan21_vace_post_process_func",
        "get_wan21_vace_pre_process_func",
        "resolve_wan21_pipeline_class_name",
        "Wan21Transformer3DModel",
        "Wan21VACETransformer3DModel",
    }

    assert expected_attrs <= package_exports


def test_wan21_transformer_factories_use_native_classes_and_forward_quant_config():
    source = _source(WAN21_COMMON)
    assert "def create_transformer_from_config(" in source
    assert "quant_config" in source
    assert "prefix" in source
    assert "Wan21Transformer3DModel(" in source
    assert "Wan21VACETransformer3DModel(" in source
    assert "quant_config=quant_config" in source
    assert "prefix=prefix" in source
    assert 'getattr(self.od_config, "quantization_config", None)' in source


def test_wan21_does_not_reject_request_default_guidance_scale_2_autofill():
    _function_def(WAN21_COMMON, "_wan21_has_unsupported_guidance_scale_2")
    prepare_forward = _function_def(WAN21_COMMON, "_prepare_common_forward")

    reject_conditions = []
    for node in ast.walk(prepare_forward):
        if not isinstance(node, ast.If):
            continue
        for statement in node.body:
            if isinstance(statement, ast.Raise) and _raises_value_error_with(
                statement,
                "Wan2.1 does not support guidance_scale_2.",
            ):
                reject_conditions.append(node.test)

    assert reject_conditions
    assert all(
        _is_name_call(condition, "_wan21_has_unsupported_guidance_scale_2")
        for condition in reject_conditions
    )


def test_wan21_i2v_requires_raw_image_for_vae_conditioning():
    source = _source(WAN21_COMMON)
    assert "raw_image is None and image_embeds is None" not in source
    assert "image_embeds alone" in source


def test_wan21_native_transformer_exposes_layerwise_offload_blocks():
    source = _source(WAN21_COMMON)
    assert "_layerwise_offload_blocks_attrs" in source
    assert "Wan21Transformer3DModel" in source


def test_wan21_make_layers_factories_accept_prefix_keyword():
    for path in (
        WAN21_TRANSFORMER,
        WAN21_DIR / "wan2_1_vace_transformer.py",
    ):
        for factory in _make_layers_factory_lambdas(path):
            parameter_names = [arg.arg for arg in factory.args.args]
            assert "prefix" in parameter_names, path


def test_wan21_defaults_attention_to_non_cudnn_sdpa_backend():
    source = _source(WAN21_COMMON)
    assert "_with_wan21_attention_defaults" in source
    assert "TORCH_SDPA_NO_CUDNN" in source
    assert "attention_config.default is not None" in source
    assert "role in per_role" in source
    assert "category in per_role" in source


def test_wan21_transformer_creation_uses_defaulted_attention_config_context():
    init_fn = _class_method_def(WAN21_COMMON, "Wan21PipelineBase", "__init__")
    matching_contexts = []
    for node in ast.walk(init_fn):
        if not isinstance(node, ast.With):
            continue
        uses_defaulted_config = any(
            _is_set_current_diffusion_config_call(item.context_expr)
            for item in node.items
        )
        if uses_defaulted_config and _contains_transformer_creation(node):
            matching_contexts.append(node)

    assert matching_contexts


def test_wan21_attention_uses_model_scoped_roles_with_category_fallback():
    source = _source(WAN21_TRANSFORMER)
    assert 'role="wan2_1.self"' in source
    assert 'role_category="self"' in source
    assert 'role="wan2_1.cross"' in source
    assert 'role_category="cross"' in source


def test_non_cudnn_sdpa_backend_is_registered():
    registry_source = _source(ATTENTION_REGISTRY)
    sdpa_source = _source(ATTENTION_SDPA)
    assert "TORCH_SDPA_NO_CUDNN" in registry_source
    assert "NoCuDNNSDPABackend" in registry_source
    assert "class NoCuDNNSDPABackend" in sdpa_source
    assert "class NoCuDNNSDPAImpl" in sdpa_source
    assert "FLASH_ATTENTION" in sdpa_source
    assert "EFFICIENT_ATTENTION" in sdpa_source
    assert "MATH" in sdpa_source
    assert "CUDNN_ATTENTION" not in sdpa_source


def test_non_cudnn_sdpa_backend_keeps_float32_fallback_non_cudnn():
    source = _source(ATTENTION_LAYER)
    assert 'self.attn_backend.get_name() == "SDPA_NO_CUDNN"' in source
    assert "fallback_backend_cls" in source
