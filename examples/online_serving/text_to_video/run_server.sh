#!/bin/bash
# Wan online serving startup script

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8098}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-}"
FLOW_SHIFT="${FLOW_SHIFT:-}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
ENABLE_CACHE_DIT_SUMMARY="${ENABLE_CACHE_DIT_SUMMARY:-0}"

echo "Starting Wan server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Flow shift: ${FLOW_SHIFT:-model default}"
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

FLOW_SHIFT_FLAG=""
if [ -n "$FLOW_SHIFT" ]; then
    FLOW_SHIFT_FLAG="--flow-shift $FLOW_SHIFT"
elif [[ "$MODEL" == *"Wan2.2"* ]]; then
    FLOW_SHIFT_FLAG="--flow-shift 5.0"
fi
if [ -n "$FLOW_SHIFT_FLAG" ]; then
    echo "Flow shift flag: $FLOW_SHIFT_FLAG"
fi

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    $BOUNDARY_RATIO_FLAG \
    $FLOW_SHIFT_FLAG \
    $CACHE_BACKEND_FLAG \
    $(if [ "$ENABLE_CACHE_DIT_SUMMARY" != "0" ]; then echo "--enable-cache-dit-summary"; fi)
