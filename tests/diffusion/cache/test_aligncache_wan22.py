# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU Wan component routing contracts; these do not establish checkpoint/GPU compatibility."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.cache.aligncache.config import AlignCacheConfig
from vllm_omni.diffusion.cache.aligncache.multicomponent import WanRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_pipeline(two_experts=True):
    timesteps = torch.arange(40, 0, -1).float()
    pipeline = SimpleNamespace(
        transformer=object(),
        transformer_2=object() if two_experts else None,
        expand_timesteps=False,
        is_dmd=False,
        record_denoise_step=lambda *a: None,
        _sample_solver="unipc",
        scheduler=SimpleNamespace(timesteps=timesteps, num_inference_steps=40, config={}),
    )
    pipeline.calls = []

    def predict_noise(**kwargs):
        pipeline.calls.append(kwargs["current_model"])
        if (runtime := getattr(pipeline, "_aligncache_runtime", None)) is not None:
            runtime.record_forward(kwargs["current_model"])
        return kwargs["hidden_states"] * 0.2 + kwargs["encoder_hidden_states"].mean()

    pipeline.predict_noise = predict_noise
    pipeline.combine_cfg_noise = lambda c, u, g, normalize: u + g * (c - u)

    def native(cfg, guidance, positive, negative, cfg_normalize):
        return pipeline.combine_cfg_noise(predict_noise(**positive), predict_noise(**negative), guidance, cfg_normalize)

    pipeline.predict_noise_maybe_with_cfg = native
    return pipeline


def bind_cpu(runtime, pipeline, x, cond, uncond, boundary=21, guidance=(5.0, 3.0)):
    # Exercise production plan building with CUDA admission metadata; execute only toy CPU tensors.
    metadata = SimpleNamespace(shape=x.shape, ndim=x.ndim, device=torch.device("cuda:0"))
    runtime.bind_schedule(
        pipeline, pipeline.scheduler.timesteps, metadata, cond, uncond, *guidance, boundary, torch.bfloat16
    )
    for child in runtime.components.values():
        if child.signature is not None:
            child.signature = replace(child.signature, device=x.device, dtype=x.dtype)


def test_profiles_are_explicit_and_independent():
    config = AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1"}})
    assert config.policy == "wan22_feedback_v1" and config.base_threshold == 0.078 and config.warmup_steps == 9
    assert AlignCacheConfig.parse().warmup_steps == 12
    assert AlignCacheConfig.parse({"align_cache": {"profile": "wan21_14b_reference_v1"}}).policy == "wan21_feedback_v1"
    with pytest.raises(ValueError, match="profile"):
        AlignCacheConfig.parse({"align_cache": {"policy": "wan22_feedback_v1"}})


@pytest.mark.parametrize("mode", ["accelerate", "trace", "force_exact", "passthrough"])
def test_component_local_state_cfg_and_counts(mode):
    runtime = WanRuntime(
        AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1", "mode": mode, "warmup_steps": 2}})
    )
    pipeline = make_pipeline()
    pipeline._aligncache_runtime = runtime
    x, cond, uncond = torch.ones(1, 16, 1, 1, 8), torch.ones(1, 2), torch.full((1, 2), 0.9)
    with runtime.request_context():
        bind_cpu(runtime, pipeline, x, cond, uncond)
        high, low = runtime.components["high"], runtime.components["low"]
        assert high.plan.evaluations == low.plan.evaluations == 20
        assert runtime.routes[19] == "high" and runtime.routes[20] == "low"
        assert high.signature.guidance_scale == 5 and low.signature.guidance_scale == 3
        for i, t in enumerate(pipeline.scheduler.timesteps):
            is_high = float(t) >= 21
            model, guidance = (pipeline.transformer, 5.0) if is_high else (pipeline.transformer_2, 3.0)
            current = x + i * 0.002
            positive = dict(hidden_states=current, current_model=model, encoder_hidden_states=cond)
            negative = dict(hidden_states=current, current_model=model, encoder_hidden_states=uncond)
            before = len(pipeline.calls)
            result = runtime.predict_noise(pipeline, i, t, guidance, positive, negative)
            if i in (0, 20):
                assert len(pipeline.calls) - before == 2
            if mode != "accelerate":
                torch.testing.assert_close(result, (current * 0.2 + 0.9) + guidance * 0.1)
        assert high.state is not low.state
        if mode == "accelerate":
            assert high.counts["fast_pairs"] > 0 and low.counts["fast_pairs"] > 0
        assert high.counts["completed_pairs"] == low.counts["completed_pairs"] == 20
    assert all(child.state is None and child.plan is None for child in runtime.components.values())
    assert runtime.routes is None
    summary = runtime.last_summary
    assert summary["completed_pairs"] == 40
    assert summary["model_forward_calls"] == len(pipeline.calls) == 2 * summary["exact_pairs"]


@pytest.mark.parametrize("boundary,active", [(0, "high"), (100, "low")])
def test_empty_component_plan(boundary, active):
    runtime = WanRuntime(AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1"}}))
    pipeline = make_pipeline()
    with runtime.request_context():
        bind_cpu(runtime, pipeline, torch.ones(1, 16, 1, 1, 8), torch.ones(1, 2), torch.zeros(1, 2), boundary)
        assert runtime.components[active].plan.evaluations == 40
        assert runtime.components["low" if active == "high" else "high"].plan is None


def test_second_component_entry_failure_unwinds_first():
    runtime = WanRuntime(AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1"}}))
    error = RuntimeError("second component failed")

    @contextmanager
    def fail():
        raise error
        yield

    runtime.components["low"].request_context = fail
    with pytest.raises(RuntimeError) as caught:
        with runtime.request_context():
            pytest.fail("entry must fail")
    assert caught.value is error
    assert not runtime.active
    assert not runtime.components["high"].active and runtime.components["high"].state is None


@pytest.mark.parametrize("mode", ["accelerate", "force_exact", "trace", "passthrough"])
def test_actual_wan22_loop_preserves_native_routing(mode):
    from types import MethodType

    from tests.diffusion.cache.aligncache_helpers import definitions

    runtime = WanRuntime(
        AlignCacheConfig.parse({"align_cache": {"profile": "wan22_t2v_reference_v1", "mode": mode, "warmup_steps": 2}})
    )
    pipeline = make_pipeline()
    pipeline._aligncache_runtime = runtime
    original_bind = runtime.bind_schedule

    def bind(*args):
        runtime.bind_schedule = original_bind
        try:
            bind_cpu(runtime, pipeline, args[2], args[3], args[4])
        finally:
            runtime.bind_schedule = bind

    runtime.bind_schedule = bind

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self):
            pass

    pipeline.progress_bar = lambda **kwargs: Progress()
    updates = []

    def scheduler(noise, t, latents, cfg):
        assert cfg
        updates.append(float(t))
        return latents - noise * 0.001

    pipeline.scheduler_step_maybe_with_cfg = scheduler
    cls = definitions(
        "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
        {"Wan22Pipeline"},
        {"set_forward_context_denoise_step_idx": lambda i: None},
        {"diffuse"},
    )["Wan22Pipeline"]
    pipeline.diffuse = MethodType(cls.diffuse, pipeline)
    with runtime.request_context():
        pipeline.diffuse(
            torch.ones(1, 16, 1, 1, 8),
            pipeline.scheduler.timesteps,
            torch.ones(1, 2),
            torch.full((1, 2), 0.9),
            5.0,
            3.0,
            21.0,
            torch.float32,
            {},
        )
    assert len(updates) == 40
    assert runtime.last_summary["model_forward_calls"] == len(pipeline.calls)
    assert len(pipeline.calls) < 80 if mode == "accelerate" else len(pipeline.calls) == 80
    assert all(child.state is None for child in runtime.components.values())


@pytest.mark.parametrize("failure", [None, "scheduler", "binding"])
def test_actual_wan_loop_runs_wan21_with_one_component(failure):
    from types import MethodType

    from tests.diffusion.cache.aligncache_helpers import definitions

    cls = definitions(
        "vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",
        {"Wan22Pipeline"},
        {"set_forward_context_denoise_step_idx": lambda i: None},
        {"diffuse"},
    )["Wan22Pipeline"]

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self):
            pass

    scheduler_error = RuntimeError("injected scheduler failure")
    results = {}
    modes = ("passthrough", "accelerate") if failure else (None, "passthrough", "force_exact", "trace", "accelerate")
    for mode in modes:
        pipeline = make_pipeline(two_experts=False)
        pipeline.progress_bar = lambda **kwargs: Progress()
        pipeline.diffuse = MethodType(cls.diffuse, pipeline)

        def native(do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize):
            assert do_true_cfg and cfg_normalize is False
            cond = pipeline.predict_noise(**positive_kwargs)
            return pipeline.combine_cfg_noise(cond, pipeline.predict_noise(**negative_kwargs), true_cfg_scale, False)

        def scheduler(noise, t, latents, cfg):
            if failure == "scheduler":
                raise scheduler_error
            if failure == "binding":
                pipeline.transformer = object()
            return latents - noise * 0.001

        pipeline.predict_noise_maybe_with_cfg = native
        pipeline.scheduler_step_maybe_with_cfg = scheduler
        runtime = None
        if mode is not None:
            runtime = WanRuntime(AlignCacheConfig.parse({"align_cache": {"mode": mode, "warmup_steps": 2}}))
            pipeline._aligncache_runtime = runtime
            original_bind = runtime.bind_schedule

            def bind(*args, runtime=runtime, original_bind=original_bind, pipeline=pipeline):
                runtime.bind_schedule = original_bind
                try:
                    bind_cpu(runtime, pipeline, args[2], args[3], args[4], args[7], args[5:7])
                finally:
                    runtime.bind_schedule = bind

            runtime.bind_schedule = bind
        context = runtime.request_context() if runtime else nullcontext()
        if failure:
            expected_error = pytest.raises(RuntimeError if failure == "scheduler" else ValueError)
        else:
            expected_error = nullcontext()
        with expected_error as caught, context:
            results[mode] = pipeline.diffuse(
                torch.ones(1, 16, 1, 1, 8),
                pipeline.scheduler.timesteps,
                torch.ones(1, 2),
                torch.full((1, 2), 0.9),
                5.0,
                5.0,
                875.0,
                torch.float32,
                {},
            )
        if failure:
            child = runtime.components["transformer"]
            assert len(pipeline.calls) == 2 and child.state is None and child.plan is None
            assert runtime.last_summary["completed_pairs"] == 1
            assert runtime.last_summary["status"] == "failed"
            if failure == "scheduler":
                assert caught.value is scheduler_error
            continue
        assert set(pipeline.calls) == {pipeline.transformer}
        assert len(pipeline.calls) < 80 if mode == "accelerate" else len(pipeline.calls) == 80
        if runtime is not None:
            assert list(runtime.components) == ["transformer"]
            assert runtime.last_summary["completed_pairs"] == 40
            assert runtime.last_summary["model_forward_calls"] == len(pipeline.calls)
    for mode in () if failure else ("passthrough", "force_exact", "trace"):
        torch.testing.assert_close(results[mode], results[None], rtol=0, atol=0)


def test_large_transformer_structural_admission_is_profile_specific():
    from vllm_omni.diffusion.cache.aligncache.adapters.wan21 import validate_transformer

    class Model:
        config = SimpleNamespace(
            in_channels=16,
            out_channels=16,
            num_layers=40,
            num_attention_heads=40,
            attention_head_dim=128,
            text_dim=4096,
            ffn_dim=13824,
            freq_dim=256,
            image_dim=None,
            added_kv_proj_dim=None,
            patch_size=(1, 2, 2),
        )

        def modules(self):
            return iter([self])

        def parameters(self):
            return iter([SimpleNamespace(dtype=torch.bfloat16, device=torch.device("cuda:0"))])

    model = Model()
    validate_transformer(model, large=True)
    with pytest.raises(ValueError, match="num_layers"):
        validate_transformer(model)
    model.config.image_dim = 1280
    with pytest.raises(ValueError, match="image_dim"):
        validate_transformer(model, large=True)
