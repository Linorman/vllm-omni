# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--last-image", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--reference-image", action="append", default=[])
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--flow-shift", type=float, required=True)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args()


def _load_image_source(source: str | None, width: int, height: int, mode: str) -> Image.Image | None:
    if source is None:
        return None
    if source.startswith("data:image"):
        _, encoded = source.split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded)))
    elif source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(source)
    image.load()
    return image.convert(mode).resize((width, height), Image.Resampling.LANCZOS)


def _load_image_sequence(source: str | None, width: int, height: int, mode: str) -> list[Image.Image] | None:
    if source is None:
        return None
    source_path = Path(source)
    if source_path.is_dir():
        suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
        paths = sorted(path for path in source_path.iterdir() if path.suffix.lower() in suffixes)
        return [_load_image_source(str(path), width, height, mode) for path in paths]
    return [_load_image_source(source, width, height, mode)]


def _set_scheduler_flow_shift(pipe: Any, flow_shift: float) -> None:
    from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler

    pipe.scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=flow_shift,
        prediction_type="flow_prediction",
    )


_DISABLE_DEVICE_MAP_VALUES = {"", "0", "false", "no", "none", "off"}


def _diffusers_load_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    device_map = os.environ.get("WAN21_DIFFUSERS_DEVICE_MAP", "balanced").strip()
    if not torch.cuda.is_available() or device_map.lower() in _DISABLE_DEVICE_MAP_VALUES:
        return kwargs

    kwargs["device_map"] = device_map
    max_memory: dict[int | str, int | str] = {
        "cpu": os.environ.get("WAN21_DIFFUSERS_MAX_CPU_MEMORY", "20GiB"),
    }
    for device_idx in range(torch.accelerator.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(device_idx)
        max_memory[device_idx] = int(free_bytes * 0.9)
    kwargs["max_memory"] = max_memory
    return kwargs


def _pipeline_execution_device(pipe: Any, fallback: str) -> torch.device:
    execution_device = getattr(pipe, "_execution_device", None)
    if execution_device is None:
        return torch.device(fallback)
    return torch.device(execution_device)


def _cast_pipeline_modules(pipe: Any, dtype: torch.dtype) -> None:
    for name in ("transformer", "transformer_2", "text_encoder", "image_encoder", "vae"):
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "to"):
            module.to(dtype=dtype)


class _IdentityFtfy:
    @staticmethod
    def fix_text(text: str) -> str:
        return text


def _ensure_wan_ftfy_fallback() -> None:
    from diffusers.pipelines.wan import pipeline_wan, pipeline_wan_i2v, pipeline_wan_vace

    for module in (pipeline_wan, pipeline_wan_i2v, pipeline_wan_vace):
        if not hasattr(module, "ftfy"):
            module.ftfy = _IdentityFtfy()


def _select_pipeline_cls(model: str):
    from diffusers import WanImageToVideoPipeline, WanPipeline, WanVACEPipeline

    normalized = model.lower()
    if "vace" in normalized:
        return WanVACEPipeline
    if "i2v" in normalized or "flf2v" in normalized:
        return WanImageToVideoPipeline
    return WanPipeline


def _extract_frames(result: Any):
    frames = getattr(result, "frames", None)
    if frames is None:
        frames = getattr(result, "videos", None)
    if frames is None and isinstance(result, (list, tuple)):
        frames = result[0]
    if isinstance(frames, (list, tuple)) and len(frames) == 1 and isinstance(frames[0], (list, tuple)):
        frames = frames[0]
    if getattr(frames, "ndim", None) == 5 and getattr(frames, "shape", (None,))[0] == 1:
        frames = frames[0]
    return frames


def _frame_count(frames: Any) -> int:
    if frames is None:
        return 0
    return len(frames)


def _write_metadata(path: str | None, *, args: argparse.Namespace, frame_count: int) -> None:
    if path is None:
        return
    payload = {
        "model": args.model,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "actual_frame_count": frame_count,
        "num_inference_steps": args.num_inference_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "conditioning_scale": args.conditioning_scale,
        "seed": args.seed,
        "image": args.image,
        "last_image": args.last_image,
        "video": args.video,
        "mask": args.mask,
        "reference_image": args.reference_image,
        "world_size": 1,
    }
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _ensure_wan_ftfy_fallback()
    pipeline_cls = _select_pipeline_cls(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = pipeline_cls.from_pretrained(args.model, **_diffusers_load_kwargs(dtype))
    _cast_pipeline_modules(pipe, dtype)
    _set_scheduler_flow_shift(pipe, args.flow_shift)
    if getattr(pipe, "hf_device_map", None) is None:
        pipe = pipe.to(device)
    device = _pipeline_execution_device(pipe, device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    call_kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
    }
    normalized_model = args.model.lower()
    image = _load_image_source(args.image, args.width, args.height, "RGB")
    last_image = _load_image_source(args.last_image, args.width, args.height, "RGB")
    if "vace" in normalized_model:
        reference_images = [
            loaded
            for loaded in (
                [_load_image_source(path, args.width, args.height, "RGB") for path in args.reference_image]
                + ([image] if image is not None else [])
                + ([last_image] if last_image is not None else [])
            )
            if loaded is not None
        ]
        video = _load_image_sequence(args.video, args.width, args.height, "RGB")
        mask = _load_image_sequence(args.mask, args.width, args.height, "L")
        if video is not None:
            call_kwargs["video"] = video
        if mask is not None:
            call_kwargs["mask"] = mask
        if reference_images:
            call_kwargs["reference_images"] = reference_images
        call_kwargs["conditioning_scale"] = args.conditioning_scale
    else:
        if image is not None:
            call_kwargs["image"] = image
        if last_image is not None:
            call_kwargs["last_image"] = last_image

    result = pipe(**call_kwargs)
    frames = _extract_frames(result)
    frame_count = _frame_count(frames)
    if frame_count == 0:
        raise RuntimeError("Diffusers Wan2.1 reference pipeline did not return video frames.")

    from diffusers.utils import export_to_video

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output), fps=args.fps)
    _write_metadata(args.metadata_output, args=args, frame_count=frame_count)


if __name__ == "__main__":
    main()
