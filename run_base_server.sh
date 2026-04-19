#!/bin/bash
# Start the base VLA server on port 8000
# Usage: ./run_base_server.sh [MODEL_PATH] [NUM_VIEWS]
# NUM_VIEWS: 1=single view (default), 2=dual view (main+wrist)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default model path if not provided
DEFAULT_MODEL="/VLA-Data/scripts/lianqing/logs/xvla/bridge_scratch/xvla_qwen3vl_2b_baseline1-02-22-13-43/ckpt-200000"
# DEFAULT_MODEL="/VLA-Data/scripts/lianqing/logs/xvla/bridge_scratch/bridge_wrist_baseline-01-29-03-06"
# DEFAULT_MODEL="/VLA-Data/scripts/lianqing/checkpoints/2toINF/X-VLA-WidowX"
MODEL_PATH="${1:-$DEFAULT_MODEL}"
NUM_VIEWS=2

echo "============================================================"
echo "  Starting VLA Base Server"
echo "============================================================"
echo "📂 Model Path: $MODEL_PATH"
echo "🔌 Port: 8000"
echo "👁️  Num Views: $NUM_VIEWS"
echo ""

# Run the deploy.py script
if command -v uv &> /dev/null; then
    CMD="uv run --active python deploy.py"
else
    CMD="python deploy.py"
fi

echo "🚀 Command: $CMD --model_path \"$MODEL_PATH\" --port 8000 --num_views $NUM_VIEWS"
$CMD --model_path "$MODEL_PATH" --port 8000 --device cuda --num_views $NUM_VIEWS
