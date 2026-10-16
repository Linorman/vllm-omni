# SPDX-License-Identifier: Apache-2.0
"""Wan21 T2V-1.3B structural allowlist and native CFG boundary."""

import json
import math

import torch

from ..state import GenerationSignature, SchedulePlan, StepContext


def validate_pipeline(pipeline, large=False):
    from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
    from vllm_omni.diffusion.models.wan2_1.pipeline_wan2_1 import Wan21Pipeline
    from vllm_omni.diffusion.models.wan2_1.wan2_1_transformer import Wan21Transformer3DModel

    if type(pipeline) is not Wan21Pipeline or type(pipeline.transformer) is not Wan21Transformer3DModel:
        raise ValueError("align_cache: requires native Wan21Pipeline / Wan21Transformer3DModel T2V")
    if type(pipeline.scheduler) is not FlowUniPCMultistepScheduler:
        raise ValueError("align_cache: requires native FlowUniPCMultistepScheduler")
    validate_transformer(pipeline.transformer, large=large)
    validate_scheduler(pipeline.scheduler)


def validate_transformer(transformer, large=False):
    expected = {
        "in_channels": 16,
        "out_channels": 16,
        "num_layers": 40 if large else 30,
        "num_attention_heads": 40 if large else 12,
        "attention_head_dim": 128,
        "text_dim": 4096,
        "ffn_dim": 13824 if large else 8960,
        "freq_dim": 256,
        "image_dim": None,
        "added_kv_proj_dim": None,
    }
    for key, value in expected.items():
        if getattr(transformer.config, key, None) != value:
            raise ValueError(f"align_cache.transformer.{key}: T2V structure requires {value}")
    if tuple(transformer.config.patch_size) != (1, 2, 2):
        raise ValueError("align_cache.transformer.patch_size: requires (1, 2, 2)")
    for module in transformer.modules():
        if any(
            getattr(module, key, None)
            for key in ("_forward_hooks", "_forward_pre_hooks", "_hf_hook", "_hook_registry", "_diffusers_hook")
        ):
            raise ValueError("align_cache: transformer hooks require separate capability validation")
        if "forward" in vars(module):
            raise ValueError("align_cache: replaced transformer forward is unsupported")
    parameter = next(transformer.parameters())
    if parameter.dtype != torch.bfloat16 or parameter.device.type != "cuda":
        raise ValueError("align_cache: native CUDA BF16 weights required")


def validate_scheduler(scheduler):
    if scheduler.config.prediction_type != "flow_prediction":
        raise ValueError("align_cache.scheduler.prediction_type: requires flow_prediction")
    scheduler_defaults = {
        "num_train_timesteps": 1000,
        "solver_order": 2,
        "solver_type": "bh2",
        "predict_x0": True,
        "use_dynamic_shifting": False,
        "thresholding": False,
        "lower_order_final": True,
        "timestep_spacing": "linspace",
        "steps_offset": 0,
        "final_sigmas_type": "zero",
    }
    for key, value in scheduler_defaults.items():
        if scheduler.config.get(key) != value:
            raise ValueError(f"align_cache.scheduler.{key}: native configuration requires {value}")
    if scheduler.solver_p is not None or scheduler.disable_corrector:
        raise ValueError("align_cache.scheduler: custom corrector/solver unsupported")
    if not math.isfinite(scheduler.config.shift) or scheduler.config.shift <= 0:
        raise ValueError("align_cache.scheduler.shift: requires finite positive value")


def bind_native_schedule(runtime, pipeline, timesteps, latents, cond, uncond, guidance, dtype, extra_kwargs):
    if timesteps is not pipeline.scheduler.timesteps or len(timesteps) != pipeline.scheduler.num_inference_steps:
        raise ValueError("align_cache.schedule: requires the scheduler's actual native time grid")
    if extra_kwargs:
        raise ValueError("align_cache: extra model conditions are unsupported")
    if latents.ndim != 5 or latents.shape[0] != 1 or dtype != torch.bfloat16 or latents.device.type != "cuda":
        raise ValueError("align_cache: requires B=1 CUDA BF16 transformer input")
    if uncond is None or cond.shape[0] != 1 or uncond.shape[0] != 1 or guidance <= 1 or not math.isfinite(guidance):
        raise ValueError("align_cache: requires stable positive/negative conditions and serial CFG > 1")
    plan = SchedulePlan(
        tuple(float(t) for t in timesteps),
        type(pipeline.scheduler).__qualname__,
        json.dumps(dict(pipeline.scheduler.config), sort_keys=True, default=str),
        id(pipeline.scheduler),
        id(timesteps),
    )
    signature = GenerationSignature(
        runtime.epoch,
        id(pipeline.transformer),
        (id(cond), id(uncond)),
        guidance,
        tuple(latents.shape),
        dtype,
        latents.device,
    )
    runtime.bind_schedule(plan, signature)


def validate_native_step(runtime, pipeline, index, t, guidance, positive, negative):
    plan = runtime.plan
    if plan is not None and plan.scheduler_id is not None:
        if (
            id(pipeline.scheduler) != plan.scheduler_id
            or id(pipeline.scheduler.timesteps) != plan.schedule_id
            or json.dumps(dict(pipeline.scheduler.config), sort_keys=True, default=str) != plan.scheduler_config
        ):
            raise ValueError("align_cache: scheduler/schedule changed within request")
    if negative is None or positive["cache_name"] != "cond" or negative["cache_name"] != "uncond":
        raise ValueError("align_cache: explicit conditional/unconditional pair required")
    if positive["current_model"] is not pipeline.transformer or negative["current_model"] is not pipeline.transformer:
        raise ValueError("align_cache: transformer binding changed")
    x = positive["hidden_states"]
    if negative["hidden_states"] is not x:
        raise ValueError("align_cache: branches must share the same latent input")
    step = StepContext(index, float(t), id(pipeline.transformer), guidance)
    conditions = (id(positive["encoder_hidden_states"]), id(negative["encoder_hidden_states"]))
    runtime._validate_step(step, x, conditions)
    return step, x, conditions


def native_pair(runtime, pipeline, index, t, guidance, positive, negative):
    step, x, conditions = validate_native_step(runtime, pipeline, index, t, guidance, positive, negative)
    return runtime.predict_pair(
        step, x, conditions, lambda branch: pipeline.predict_noise(**(positive if branch == "cond" else negative))
    )
