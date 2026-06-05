# Wan2.1 Accuracy Model Assets

## Release-gated GPU validation set

Download these checkpoints before running the Wan2.1 accuracy suite:

| Purpose | Repository |
| --- | --- |
| T2V small model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| T2V 14B model | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| I2V 480P image conditioning | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` |
| I2V 720P image conditioning | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` |
| FLF2V first/last frame conditioning | `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers` |
| VACE small reference-image conditioning | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` |
| VACE 14B reference-image conditioning | `Wan-AI/Wan2.1-VACE-14B-diffusers` |

## Test input assets

Set these environment variables for GPU tests:

```bash
export WAN21_IMAGE_SOURCE=/data/test-assets/wan21/input.png
export WAN21_LAST_IMAGE_SOURCE=/data/test-assets/wan21/last.png
```

Buildkite creates deterministic synthetic RGB defaults at
`/tmp/wan21-test-assets/input.png` and `/tmp/wan21-test-assets/last.png` when
these variables are not supplied. Manual GPU runs can still point
`WAN21_IMAGE_SOURCE` and `WAN21_LAST_IMAGE_SOURCE` at explicit local or remote
assets.

`WAN21_IMAGE_SOURCE` is required for I2V, FLF2V, and VACE reference-image cases.
`WAN21_LAST_IMAGE_SOURCE` is required for FLF2V. Use RGB images with visible
subject and lighting differences. Use stable checked-in synthetic assets for CI
if licensing prevents storing real examples.

The Diffusers reference runner also accepts VACE-specific conditioning assets:
`--video` and `--mask` can point to a single image or a directory of ordered image
frames, and `--reference-image` can be repeated for additional reference images.

## Download commands

```bash
base=/data/hf-models/wan21-diffusers
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --local-dir "$base/Wan2.1-T2V-1.3B-Diffusers"
hf download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers --local-dir "$base/Wan2.1-I2V-14B-480P-Diffusers"
hf download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers --local-dir "$base/Wan2.1-FLF2V-14B-720P-diffusers"
hf download Wan-AI/Wan2.1-VACE-1.3B-diffusers --local-dir "$base/Wan2.1-VACE-1.3B-diffusers"
hf download Wan-AI/Wan2.1-T2V-14B-Diffusers --local-dir "$base/Wan2.1-T2V-14B-Diffusers"
hf download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers --local-dir "$base/Wan2.1-I2V-14B-720P-Diffusers"
hf download Wan-AI/Wan2.1-VACE-14B-diffusers --local-dir "$base/Wan2.1-VACE-14B-diffusers"
```
