# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import PIL.Image
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WAN21_DIR = PROJECT_ROOT / "vllm_omni" / "diffusion" / "models" / "wan2_1"
ATTENTION_LAYER = PROJECT_ROOT / "vllm_omni" / "diffusion" / "attention" / "layer.py"
ATTENTION_REGISTRY = PROJECT_ROOT / "vllm_omni" / "diffusion" / "attention" / "backends" / "registry.py"
ATTENTION_SDPA = PROJECT_ROOT / "vllm_omni" / "diffusion" / "attention" / "backends" / "sdpa.py"
CACHE_BACKEND = PROJECT_ROOT / "vllm_omni" / "diffusion" / "cache" / "cache_dit_backend.py"
REGISTRY = PROJECT_ROOT / "vllm_omni" / "diffusion" / "registry.py"
MODEL_METADATA = PROJECT_ROOT / "vllm_omni" / "diffusion" / "model_metadata.py"
PIPELINE_PARALLEL = PROJECT_ROOT / "vllm_omni" / "diffusion" / "distributed" / "pipeline_parallel.py"
WAN21_COMMON = WAN21_DIR / "pipeline_wan2_1_common.py"
WAN21_TRANSFORMER = WAN21_DIR / "wan2_1_transformer.py"
WAN21_VACE_TRANSFORMER = WAN21_DIR / "wan2_1_vace_transformer.py"


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.ndim = len(shape)
        self.repeat_args: tuple[int, ...] | None = None

    def repeat(self, *args: int):
        self.repeat_args = args
        return FakeTensor((self.shape[0] * args[0], *self.shape[1:]))


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


def _dict_entry_node(path: Path, dict_name: str, key: str) -> ast.AST:
    tree = ast.parse(_source(path))
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == dict_name for target in node.targets
        ):
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == dict_name:
            value = node.value
        if value is None:
            continue
        assert isinstance(value, ast.Dict)
        for dict_key, dict_value in zip(value.keys, value.values):
            if isinstance(dict_key, ast.Constant) and dict_key.value == key:
                return dict_value
    raise AssertionError(f"{dict_name}[{key!r}] not found in {path}")


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


def _exec_common_functions(*names: str) -> dict[str, Any]:
    tree = ast.parse(_source(WAN21_COMMON))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    module = ast.Module(body=[functions[name] for name in names], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "PIL": PIL,
        "torch": SimpleNamespace(Tensor=FakeTensor),
    }
    exec(compile(module, str(WAN21_COMMON), "exec"), namespace)
    return namespace


def _class_method_def(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(_source(path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == method_name:
                return member
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _class_def(path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(_source(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found in {path}")


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
        "pipeline_wan2_1_flf2v.py": ("Wan21FLF2VPipeline", "Wan21FLF2VPipelineBase"),
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
    assert '"Wan21FLF2VPipeline": enable_cache_for_wan21' in source
    assert '"Wan21VACEPipeline": enable_cache_for_wan21' in source
    assert '"Wan21Pipeline": enable_cache_for_wan22' not in source
    assert '"Wan21I2VPipeline": enable_cache_for_wan22' not in source
    assert '"Wan21FLF2VPipeline": enable_cache_for_wan22' not in source
    assert '"Wan21VACEPipeline": enable_cache_for_wan22' not in source


def test_wan21_transformer_config_forwards_i2v_added_kv_projection_dim():
    source = ast.get_source_segment(
        _source(WAN21_COMMON),
        _function_def(WAN21_COMMON, "_make_wan_transformer_kwargs"),
    )
    assert source is not None
    assert '"image_dim"' in source
    assert '"added_kv_proj_dim"' in source


def test_wan21_registered_process_functions_resolve():
    models = _assigned_dict(REGISTRY, "_DIFFUSION_MODELS")
    pre_process_funcs = _assigned_dict(REGISTRY, "_DIFFUSION_PRE_PROCESS_FUNCS")
    post_process_funcs = _assigned_dict(REGISTRY, "_DIFFUSION_POST_PROCESS_FUNCS")
    for arch in ("Wan21Pipeline", "Wan21I2VPipeline", "Wan21FLF2VPipeline", "Wan21VACEPipeline"):
        _, mod_relname, _ = models[arch]
        module_attrs = _module_attrs(WAN21_DIR / f"{mod_relname}.py")
        process_func_names = {
            pre_process_funcs[arch],
            post_process_funcs[arch],
        }

        for func_name in process_func_names:
            assert func_name in module_attrs, (arch, func_name)


def test_wan21_flf2v_pipeline_is_registered():
    models = _assigned_dict(REGISTRY, "_DIFFUSION_MODELS")
    pre_process_funcs = _assigned_dict(REGISTRY, "_DIFFUSION_PRE_PROCESS_FUNCS")
    post_process_funcs = _assigned_dict(REGISTRY, "_DIFFUSION_POST_PROCESS_FUNCS")

    assert models["Wan21FLF2VPipeline"] == (
        "wan2_1",
        "pipeline_wan2_1_flf2v",
        "Wan21FLF2VPipeline",
    )
    assert pre_process_funcs["Wan21FLF2VPipeline"] == "get_wan21_flf2v_pre_process_func"
    assert post_process_funcs["Wan21FLF2VPipeline"] == "get_wan21_flf2v_post_process_func"


def test_wan21_package_exports_match_registered_entrypoints():
    package_exports = set(_assigned_literal(WAN21_DIR / "__init__.py", "__all__"))
    expected_attrs = {
        "WAN21_FLF2V_PIPELINE",
        "Wan21Pipeline",
        "get_wan21_post_process_func",
        "get_wan21_pre_process_func",
        "Wan21I2VPipeline",
        "get_wan21_i2v_post_process_func",
        "get_wan21_i2v_pre_process_func",
        "Wan21FLF2VPipeline",
        "get_wan21_flf2v_post_process_func",
        "get_wan21_flf2v_pre_process_func",
        "Wan21VACEPipeline",
        "get_wan21_vace_post_process_func",
        "get_wan21_vace_pre_process_func",
        "resolve_wan21_pipeline_class_name",
        "Wan21Transformer3DModel",
        "Wan21VACETransformer3DModel",
    }

    assert expected_attrs <= package_exports


def test_wan21_flf2v_metadata_supports_two_image_inputs():
    entry = _dict_entry_node(MODEL_METADATA, "_DIFFUSION_MODEL_METADATA", "Wan21FLF2VPipeline")
    assert isinstance(entry, ast.Call)
    assert isinstance(entry.func, ast.Name)
    assert entry.func.id == "DiffusionModelMetadata"
    kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in entry.keywords}

    assert kwargs["supports_multimodal_inputs"] is True
    assert kwargs["max_multimodal_image_inputs"] == 2


def test_wan21_flf2v_common_base_extends_i2v_and_requires_last_image_conditioning():
    tree = ast.parse(_source(WAN21_COMMON))
    classes = {
        node.name: [base.id for base in node.bases if isinstance(base, ast.Name)]
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    assert classes["Wan21FLF2VPipelineBase"] == ["Wan21I2VPipelineBase"]

    prepare = _class_method_def(WAN21_COMMON, "Wan21FLF2VPipelineBase", "prepare_i2v_latents")
    forward = _class_method_def(WAN21_COMMON, "Wan21FLF2VPipelineBase", "forward")
    prepare_source = ast.get_source_segment(_source(WAN21_COMMON), prepare)
    forward_source = ast.get_source_segment(_source(WAN21_COMMON), forward)

    assert "last_image: torch.Tensor | None = None" in prepare_source
    assert "Wan2.1 FLF2V requires a last image" in prepare_source
    assert "raw_last_image" in forward_source
    assert "self.encode_image([image, last_image], self.device)" in forward_source
    assert "last_image=last_image_tensor" in forward_source


def test_wan21_flf2v_preprocessor_normalizes_two_image_forms():
    source = _source(WAN21_COMMON)
    assert "def _normalize_wan21_flf2v_images(" in source
    assert "def _prepare_wan21_flf2v_image_embeds(" in source
    assert 'multi_modal_data.get("last_image")' in source
    assert 'multi_modal_data["image"] = first_image' in source
    assert 'multi_modal_data["last_image"] = last_image' in source
    assert "Wan2.1 FLF2V requires both first and last images" in source
    assert "def get_wan21_flf2v_pre_process_func(" in source
    assert "def get_wan21_flf2v_post_process_func(" in source


def test_wan21_flf2v_image_embeds_helper_accepts_unexpanded_and_expanded_inputs():
    helper = _exec_common_functions("_prepare_wan21_flf2v_image_embeds")["_prepare_wan21_flf2v_image_embeds"]
    prompt_batch_size = 3
    unexpanded = FakeTensor((2, 4, 5))
    expanded = FakeTensor((2 * prompt_batch_size, 4, 5))

    result = helper(unexpanded, prompt_batch_size)
    assert result.shape == expanded.shape
    assert unexpanded.repeat_args == (prompt_batch_size, 1, 1)
    assert helper(expanded, prompt_batch_size) is expanded


def test_wan21_flf2v_image_embeds_helper_rejects_ambiguous_batch_shapes():
    helper = _exec_common_functions("_prepare_wan21_flf2v_image_embeds")["_prepare_wan21_flf2v_image_embeds"]

    with pytest.raises(ValueError, match=r"Wan2\.1 FLF2V image_embeds must have shape"):
        helper(FakeTensor((3, 8, 5)), prompt_batch_size=3)
    with pytest.raises(ValueError, match=r"Wan2\.1 FLF2V image_embeds must have shape"):
        helper(FakeTensor((3, 4, 5)), prompt_batch_size=3)
    with pytest.raises(ValueError, match=r"Wan2\.1 FLF2V image_embeds must be a 3D tensor"):
        helper(FakeTensor((2, 4)), prompt_batch_size=3)


def test_wan21_flf2v_image_list_normalizer_rejects_conflicts_and_bad_lengths():
    namespace = _exec_common_functions("_load_pil_image", "_normalize_wan21_flf2v_images")
    normalize = namespace["_normalize_wan21_flf2v_images"]
    first = PIL.Image.new("RGB", (8, 8))
    last = PIL.Image.new("RGB", (8, 8))

    list_form = {"image": [first, last]}
    normalized_first, normalized_last = normalize(list_form)
    assert isinstance(normalized_first, PIL.Image.Image)
    assert isinstance(normalized_last, PIL.Image.Image)
    assert list_form["image"] is normalized_first
    assert list_form["last_image"] is normalized_last

    dict_form = {"image": first, "last_image": last}
    normalized_first, normalized_last = normalize(dict_form)
    assert dict_form["image"] is normalized_first
    assert dict_form["last_image"] is normalized_last

    with pytest.raises(ValueError, match=r"either image as \[first, last\] or last_image"):
        normalize({"image": [first, last], "last_image": last})
    with pytest.raises(ValueError, match=r"exactly two images"):
        normalize({"image": [first]})
    with pytest.raises(ValueError, match=r"exactly two images"):
        normalize({"image": [first, last, first]})


def test_wan21_pipeline_base_inherits_pipeline_parallel_before_cfg():
    source = _source(WAN21_COMMON)
    assert "AsyncLatents" in source
    assert "PipelineParallelMixin" in source
    assert "IntermediateTensors" in source
    assert "set_forward_context_denoise_step_idx" in source

    cls = _class_def(WAN21_COMMON, "Wan21PipelineBase")
    bases = [base.id for base in cls.bases if isinstance(base, ast.Name)]
    assert "PipelineParallelMixin" in bases
    assert "CFGParallelMixin" in bases
    assert bases.index("PipelineParallelMixin") < bases.index("CFGParallelMixin")


def test_wan21_predict_noise_preserves_intermediate_tensors_for_pipeline_parallel():
    predict_noise = _class_method_def(WAN21_COMMON, "Wan21PipelineBase", "predict_noise")
    source = ast.get_source_segment(_source(WAN21_COMMON), predict_noise)
    assert source is not None

    assert "torch.Tensor | IntermediateTensors" in ast.unparse(predict_noise.returns)
    assert "isinstance(result, IntermediateTensors)" in source
    assert "return result" in source


def test_pipeline_parallel_vae_decode_wrapper_is_idempotent_for_wan21_inheritance():
    source = _source(PIPELINE_PARALLEL)
    assert "_vllm_omni_pp_wrapped" in source
    assert 'getattr(orig_decode, "_vllm_omni_pp_wrapped", False)' in source


@pytest.mark.parametrize(
    ("class_name", "needs_condition"),
    [
        ("Wan21PipelineBase", False),
        ("Wan21I2VPipelineBase", True),
    ],
)
def test_wan21_diffuse_methods_use_pp_aware_denoise_wrappers(class_name: str, needs_condition: bool):
    diffuse = _class_method_def(WAN21_COMMON, class_name, "diffuse")
    source = ast.get_source_segment(_source(WAN21_COMMON), diffuse)
    assert source is not None

    assert "torch.Tensor | AsyncLatents" in ast.unparse(diffuse.returns)
    assert "predict_noise_maybe_with_cfg" in source
    assert "scheduler_step_maybe_with_cfg" in source
    assert "for step_idx, t in enumerate(timesteps)" in source
    assert "set_forward_context_denoise_step_idx(step_idx)" in source
    assert "set_forward_context_denoise_step_idx(None)" in source
    if needs_condition:
        assert "torch.cat([latents, condition], dim=1)" in source
    else:
        assert "latents.to(dtype)" in source


def test_wan21_vace_transformer_is_pipeline_parallel_aware():
    source = _source(WAN21_VACE_TRANSFORMER)
    forward = _class_method_def(WAN21_VACE_TRANSFORMER, "Wan21VACETransformer3DModel", "forward")
    forward_source = ast.get_source_segment(source, forward)
    assert forward_source is not None

    assert "IntermediateTensors" in source
    assert "is_pipeline_first_stage" in source
    assert "is_pipeline_last_stage" in source
    assert "intermediate_tensors: IntermediateTensors | None = None" in forward_source
    assert "torch.Tensor | Transformer2DModelOutput | IntermediateTensors" in ast.unparse(forward.returns)
    assert "intermediate_tensors must be provided for non-first PP stages" in forward_source
    assert 'intermediate_tensors["hidden_states"]' in forward_source
    assert 'intermediate_tensors[f"vace_hint_{i}"]' in forward_source
    assert "vace_hints must be provided for non-first VACE PP stages" in forward_source
    assert "self.blocks[self.start_layer : self.end_layer]" in forward_source
    assert "return IntermediateTensors(tensors)" in forward_source
    assert 'f"vace_hint_{i}"' in forward_source


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
    assert all(_is_name_call(condition, "_wan21_has_unsupported_guidance_scale_2") for condition in reject_conditions)


def test_wan21_prepare_forward_resets_scheduler_begin_index_like_diffusers():
    prepare_forward = _function_def(WAN21_COMMON, "_prepare_common_forward")
    source = ast.get_source_segment(_source(WAN21_COMMON), prepare_forward) or ""

    assert "self.scheduler.set_timesteps(num_steps, device=self.device)" in source
    assert "self.scheduler.set_begin_index(0)" in source
    assert source.index("self.scheduler.set_timesteps(num_steps, device=self.device)") < source.index(
        "self.scheduler.set_begin_index(0)"
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


def test_wan21_does_not_override_attention_backend_defaults():
    source = _source(WAN21_COMMON)
    assert "_with_wan21_attention_defaults" not in source
    assert "WAN21_DEFAULT_ATTENTION_BACKEND" not in source
    assert "TORCH_SDPA_NO_CUDNN" not in source


def test_wan21_transformer_creation_uses_current_attention_config_context():
    init_fn = _class_method_def(WAN21_COMMON, "Wan21PipelineBase", "__init__")
    matching_contexts = []
    for node in ast.walk(init_fn):
        if not isinstance(node, ast.With):
            continue
        uses_defaulted_config = any(_is_set_current_diffusion_config_call(item.context_expr) for item in node.items)
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
