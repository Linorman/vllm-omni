#!/bin/bash
# Wan image-to-video curl example using the async video job API.

set -euo pipefail

MODEL="${MODEL:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
INPUT_IMAGE="${INPUT_IMAGE:-../../offline_inference/image_to_video/qwen-bear.png}"
LAST_INPUT_IMAGE="${LAST_INPUT_IMAGE:-}"
BASE_URL="${BASE_URL:-http://localhost:8099}"
PROMPT="${PROMPT:-A bear playing with yarn, smooth motion}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
SIZE="${SIZE:-832x480}"
CLIP_SECONDS="${CLIP_SECONDS:-2}"
FPS="${FPS:-16}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
SEED="${SEED:-42}"

IS_WAN21=0
if [[ "${MODEL}" == *"Wan2.1"* || "${MODEL}" == *"wan2.1"* ]]; then
  IS_WAN21=1
fi

if [ "${IS_WAN21}" = "1" ]; then
  GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
  if [ -z "${FLOW_SHIFT:-}" ]; then
    if [[ "${MODEL}" == *"FLF2V"* || "${MODEL}" == *"flf2v"* ]]; then
      FLOW_SHIFT="16.0"
    elif [[ ( "${MODEL}" == *"I2V"* || "${MODEL}" == *"i2v"* ) && ( "${MODEL}" == *"720P"* || "${MODEL}" == *"720p"* ) ]]; then
      FLOW_SHIFT="5.0"
    else
      FLOW_SHIFT="3.0"
    fi
  fi
else
  GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
  GUIDANCE_SCALE_2="${GUIDANCE_SCALE_2:-1.0}"
  BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
  FLOW_SHIFT="${FLOW_SHIFT:-12.0}"
fi

if [ -z "${OUTPUT_PATH:-}" ]; then
  if [ "${IS_WAN21}" = "1" ]; then
    OUTPUT_PATH="wan21_i2v_output.mp4"
  else
    OUTPUT_PATH="wan22_i2v_output.mp4"
  fi
fi

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Input image not found: $INPUT_IMAGE"
    exit 1
fi
if [ -n "${LAST_INPUT_IMAGE}" ] && [ ! -f "${LAST_INPUT_IMAGE}" ]; then
    echo "Last input image not found: ${LAST_INPUT_IMAGE}"
    exit 1
fi

create_cmd=(
  curl -sS -X POST "${BASE_URL}/v1/videos"
  -H "Accept: application/json"
  -F "model=${MODEL}"
  -F "prompt=${PROMPT}"
  -F "input_reference=@${INPUT_IMAGE}"
  -F "seconds=${CLIP_SECONDS}"
  -F "size=${SIZE}"
  -F "fps=${FPS}"
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}"
  -F "guidance_scale=${GUIDANCE_SCALE}"
  -F "flow_shift=${FLOW_SHIFT}"
  -F "seed=${SEED}"
)

if [ -n "${LAST_INPUT_IMAGE}" ]; then
  create_cmd+=(-F "last_input_reference=@${LAST_INPUT_IMAGE}")
fi

if [ "${IS_WAN21}" != "1" ]; then
  create_cmd+=(
    -F "guidance_scale_2=${GUIDANCE_SCALE_2}"
    -F "boundary_ratio=${BOUNDARY_RATIO}"
  )
fi

if [ -n "${NEGATIVE_PROMPT}" ]; then
  create_cmd+=(-F "negative_prompt=${NEGATIVE_PROMPT}")
fi

if [ -n "${SAMPLE_SOLVER}" ]; then
  create_cmd+=(-F "extra_params={\"sample_solver\":\"${SAMPLE_SOLVER}\"}")
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
