from __future__ import annotations

MODEL_T2V_13B = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_T2V_14B = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
MODEL_I2V_480P = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
MODEL_I2V_720P = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
MODEL_FLF2V_720P = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
MODEL_VACE_13B = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
MODEL_VACE_14B = "Wan-AI/Wan2.1-VACE-14B-diffusers"

RELEASE_GATE_MODELS = (
    MODEL_T2V_13B,
    MODEL_T2V_14B,
    MODEL_I2V_480P,
    MODEL_I2V_720P,
    MODEL_FLF2V_720P,
    MODEL_VACE_13B,
    MODEL_VACE_14B,
)

SIZE = "320x480"
WIDTH = 320
HEIGHT = 480
FPS = 8
NUM_FRAMES = 5
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 5.0
CONDITIONING_SCALE = 1.0
SEED = 42
SSIM_THRESHOLD = 0.90
PSNR_THRESHOLD = 24.0

FLOW_SHIFT_T2V = 3.0
FLOW_SHIFT_I2V_480P = 3.0
FLOW_SHIFT_I2V_720P = 5.0
FLOW_SHIFT_FLF2V_720P = 16.0
FLOW_SHIFT_VACE = 3.0

FLOW_SHIFT_BY_MODEL = {
    MODEL_T2V_13B: FLOW_SHIFT_T2V,
    MODEL_T2V_14B: FLOW_SHIFT_T2V,
    MODEL_I2V_480P: FLOW_SHIFT_I2V_480P,
    MODEL_I2V_720P: FLOW_SHIFT_I2V_720P,
    MODEL_FLF2V_720P: FLOW_SHIFT_FLF2V_720P,
    MODEL_VACE_13B: FLOW_SHIFT_VACE,
    MODEL_VACE_14B: FLOW_SHIFT_VACE,
}

PROMPT = "A small robot carefully watering bright flowers in a quiet greenhouse."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text, distorted"
I2V_PROMPT = "A camera slowly pushes in on a bright toy robot beside greenhouse flowers."
FLF2V_PROMPT = "A smooth transition from a bright toy robot to a blue evening greenhouse scene."
VACE_PROMPT = "A reference-guided greenhouse shot with soft natural motion and clear subject detail."

PROMPT_BY_MODEL = {
    MODEL_T2V_13B: PROMPT,
    MODEL_T2V_14B: PROMPT,
    MODEL_I2V_480P: I2V_PROMPT,
    MODEL_I2V_720P: I2V_PROMPT,
    MODEL_FLF2V_720P: FLF2V_PROMPT,
    MODEL_VACE_13B: VACE_PROMPT,
    MODEL_VACE_14B: VACE_PROMPT,
}
