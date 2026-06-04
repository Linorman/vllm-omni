#!/bin/bash
# Wan online serving startup script

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8098}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
ENABLE_CACHE_DIT_SUMMARY="${ENABLE_CACHE_DIT_SUMMARY:-0}"

echo "Starting Wan server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Flow shift: $FLOW_SHIFT"
echo "Cache backend: $CACHE_BACKEND"
if [ "$ENABLE_CACHE_DIT_SUMMARY" != "0" ]; then
    echo "Cache-DiT summary: enabled"
fi

CACHE_BACKEND_FLAG=""
if [ "$CACHE_BACKEND" != "none" ]; then
    CACHE_BACKEND_FLAG="--cache-backend $CACHE_BACKEND"
fi

BOUNDARY_RATIO_FLAG=""
if [ -n "$BOUNDARY_RATIO" ]; then
    BOUNDARY_RATIO_FLAG="--boundary-ratio $BOUNDARY_RATIO"
elif [[ "$MODEL" == *"Wan2.2"* ]]; then
    BOUNDARY_RATIO_FLAG="--boundary-ratio 0.875"
fi
if [ -n "$BOUNDARY_RATIO_FLAG" ]; then
    echo "Boundary ratio flag: $BOUNDARY_RATIO_FLAG"
fi

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    $BOUNDARY_RATIO_FLAG \
    --flow-shift "$FLOW_SHIFT" \
    $CACHE_BACKEND_FLAG \
    $(if [ "$ENABLE_CACHE_DIT_SUMMARY" != "0" ]; then echo "--enable-cache-dit-summary"; fi)
