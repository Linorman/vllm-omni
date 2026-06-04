#!/bin/bash
# Wan text-to-video curl example using the async video job API.

set -euo pipefail

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
BASE_URL="${BASE_URL:-http://localhost:8098}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
PROMPT="${PROMPT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走}"
SIZE="${SIZE:-832x480}"
CLIP_SECONDS="${CLIP_SECONDS:-2}"
FPS="${FPS:-16}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
SEED="${SEED:-42}"

IS_WAN21=0
if [[ "${MODEL}" == *"Wan2.1"* || "${MODEL}" == *"wan2.1"* ]]; then
  IS_WAN21=1
fi

if [ -z "${OUTPUT_PATH:-}" ]; then
  if [ "${IS_WAN21}" = "1" ]; then
    OUTPUT_PATH="wan21_output.mp4"
  else
    OUTPUT_PATH="wan22_output.mp4"
  fi
fi

create_cmd=(
  curl -sS -X POST "${BASE_URL}/v1/videos"
  -H "Accept: application/json"
  -F "model=${MODEL}"
  -F "prompt=${PROMPT}"
  -F "seconds=${CLIP_SECONDS}"
  -F "size=${SIZE}"
  -F "negative_prompt=${NEGATIVE_PROMPT}"
  -F "fps=${FPS}"
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}"
  -F "guidance_scale=${GUIDANCE_SCALE}"
  -F "flow_shift=${FLOW_SHIFT}"
  -F "seed=${SEED}"
)

if [ "${IS_WAN21}" != "1" ]; then
  GUIDANCE_SCALE_2="${GUIDANCE_SCALE_2:-4.0}"
  BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
  create_cmd+=(
    -F "guidance_scale_2=${GUIDANCE_SCALE_2}"
    -F "boundary_ratio=${BOUNDARY_RATIO}"
  )
fi

create_response="$("${create_cmd[@]}")"

video_id="$(echo "${create_response}" | jq -r '.id')"
if [ -z "${video_id}" ] || [ "${video_id}" = "null" ]; then
  echo "Failed to create video job:"
  echo "${create_response}" | jq .
  exit 1
fi

echo "Created video job ${video_id}"
echo "${create_response}" | jq .

while true; do
  status_response="$(curl -sS "${BASE_URL}/v1/videos/${video_id}")"
  status="$(echo "${status_response}" | jq -r '.status')"

  case "${status}" in
    queued|in_progress)
      echo "Video job ${video_id} status: ${status}"
      sleep "${POLL_INTERVAL}"
      ;;
    completed)
      echo "${status_response}" | jq .
      break
      ;;
    failed)
      echo "Video generation failed:"
      echo "${status_response}" | jq .
      exit 1
      ;;
    *)
      echo "Unexpected status response:"
      echo "${status_response}" | jq .
      exit 1
      ;;
  esac
done

curl -sS -L "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_PATH}"
echo "Saved video to ${OUTPUT_PATH}"
