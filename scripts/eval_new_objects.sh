#!/bin/bash
# ============================================================
# New Object Evaluation Script for WidowX Tasks
# Tests newly imported objects (Kingston USB, etc.)
# ============================================================

set -e  # Exit on error

# Navigate to project root
cd /VLA-Data/scripts/lingyiran/x-vla-main
source .venv/bin/activate

# ============================================================
# Configuration
# ============================================================

# Default parameters (can be overridden via command line)
CONNECTION_INFO="${1:-logs/info.json}"
NUM_EPISODES="${2:-24}"
MAX_STEPS="${3:-120}"
SEED="${4:-20260116}"
OUTPUT_DIR="${5:-logs/new_objects_eval}"
DOMAIN_ID="${6:-0}"

# Task to run (default: kingston_usb_9grid)
TASK="${7:-kingston_usb_9grid}"

# ============================================================
# Help
# ============================================================

show_help() {
    echo "Usage: $0 [CONNECTION_INFO] [NUM_EPISODES] [MAX_STEPS] [SEED] [OUTPUT_DIR] [DOMAIN_ID] [TASK]"
    echo ""
    echo "Arguments:"
    echo "  CONNECTION_INFO   Path to server info.json (default: logs/info.json)"
    echo "  NUM_EPISODES      Number of episodes per task (default: 24)"
    echo "  MAX_STEPS         Maximum steps per episode (default: 120)"
    echo "  SEED              Base random seed (default: 20260116)"
    echo "  OUTPUT_DIR        Output directory (default: logs/new_objects_eval)"
    echo "  DOMAIN_ID         Domain ID for model (default: 0)"
    echo "  TASK              Task key from task_configs_new_objects.json (default: kingston_usb_9grid)"
    echo ""
    echo "Example:"
    echo "  $0 logs/info.json 24 120 42 logs/test kingston_usb_9grid"
    echo ""
    echo "Available tasks (see task_configs_new_objects.json):"
    echo "  - kingston_usb_9grid: Test Kingston USB model (9-grid positions)"
    exit 0
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
fi

# ============================================================
# Pre-flight checks
# ============================================================

echo "============================================================"
echo "🚀 New Object Evaluation Script"
echo "============================================================"
echo ""
echo "📋 Configuration:"
echo "   Connection Info: ${CONNECTION_INFO}"
echo "   Episodes/Task:   ${NUM_EPISODES}"
echo "   Max Steps:       ${MAX_STEPS}"
echo "   Seed:            ${SEED}"
echo "   Output Dir:      ${OUTPUT_DIR}"
echo "   Domain ID:       ${DOMAIN_ID}"
echo "   Task:            ${TASK}"
echo ""

# Check if connection info exists
if [[ ! -f "${CONNECTION_INFO}" ]]; then
    echo "❌ Error: Connection info file not found: ${CONNECTION_INFO}"
    echo "   Please start the XVLA server first and provide the correct path."
    exit 1
fi

echo "✅ Connection info found."

# Check if task config exists
TASK_CONFIG="evaluation/simpler/WidowX/task_configs_new_objects.json"
if [[ ! -f "${TASK_CONFIG}" ]]; then
    echo "❌ Error: Task config file not found: ${TASK_CONFIG}"
    echo "   Please ensure task_configs_new_objects.json exists."
    exit 1
fi

echo "✅ Task config found."
echo ""

# ============================================================
# Run evaluation
# ============================================================

echo "🔄 Starting new object evaluation..."
echo ""

uv run -m evaluation.simpler.WidowX.client_new_objects \
    --task "${TASK}" \
    --connection_info "${CONNECTION_INFO}" \
    --num_episodes ${NUM_EPISODES} \
    --max_steps ${MAX_STEPS} \
    --seed ${SEED} \
    --output_dir "${OUTPUT_DIR}/${TASK}" \
    --domain_id ${DOMAIN_ID}

echo ""
echo "============================================================"
echo "✅ New object evaluation completed!"
echo "   Results saved to: ${OUTPUT_DIR}/${TASK}"
echo "============================================================"
