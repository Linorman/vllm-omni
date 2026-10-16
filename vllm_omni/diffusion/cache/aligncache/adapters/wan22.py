# SPDX-License-Identifier: Apache-2.0
"""Native Wan2.2 T2V-A14B admission. I2V/TI2V remain separate gates."""

from .wan21 import validate_scheduler, validate_transformer


def validate_pipeline(pipeline):
    from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

    if type(pipeline) is not Wan22Pipeline or pipeline.expand_timesteps or getattr(pipeline, "is_dmd", False):
        raise ValueError("align_cache: Wan2.2 requires native T2V-A14B, not I2V/TI2V or subclasses")
    for model in (pipeline.transformer, pipeline.transformer_2):
        if type(model) is not WanTransformer3DModel:
            raise ValueError("align_cache: Wan2.2 requires both native high/low noise transformers")
        validate_transformer(model, large=True)
    if pipeline.transformer is pipeline.transformer_2:
        raise ValueError("align_cache: Wan2.2 components must have distinct identities")
    if type(pipeline.scheduler) is not FlowUniPCMultistepScheduler:
        raise ValueError("align_cache: Wan2.2 requires FlowUniPCMultistepScheduler")
    validate_scheduler(pipeline.scheduler)
