#!/bin/bash
# ============================================================
# Variant Batch Evaluation Script for WidowX Tasks
# Runs random_camera and random_light variants (2 rounds total)
# ============================================================

set -e  # Exit on error

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
source .venv/bin/activate

# ============================================================
# Configuration
# ============================================================

# Default parameters (can be overridden via command line)
CONNECTION_INFO="${1:-logs/info.json}"
NUM_EPISODES="${2:-24}"
MAX_STEPS="${3:-120}"
SEED="${4:-20260114}"
OUTPUT_DIR="${5:-logs/variant_eval}"
DOMAIN_ID="${6:-0}"

# Variants to run (all by default)
# To run specific variants, modify this line or pass via command line
# Available: random_camera, random_light, random_ee_pose, 
#            random_instruction_verb, random_instruction_adjective, random_instruction_noun, 
#            sim2real, multi_object, table_color
# VARIANTS="random_camera random_light random_ee_pose random_instruction_verb random_instruction_adjective random_instruction_noun sim2real multi_object table_color"
# VARIANTS="random_camera"
# 修改脚本中的VARIANTS变量为:
VARIANTS="table_color"

# bash eval_variant_tasks.sh logs/info.json

# ============================================================
# Help
# ============================================================

show_help() {
    echo "Usage: $0 [CONNECTION_INFO] [NUM_EPISODES] [MAX_STEPS] [SEED] [OUTPUT_DIR] [DOMAIN_ID]"
    echo ""
    echo "Arguments:"
    echo "  CONNECTION_INFO   Path to server info.json (default: logs/info.json)"
    echo "  NUM_EPISODES      Number of episodes per task (default: 24)"
    echo "  MAX_STEPS         Maximum steps per episode (default: 120)"
    echo "  SEED              Base random seed (default: 20260114)"
    echo "  OUTPUT_DIR        Output directory (default: logs/variant_eval)"
    echo "  DOMAIN_ID         Domain ID for model (default: 0)"
    echo ""
    echo "Example:"
    echo "  $0 logs/info.json 24 120 42"
    echo ""
    echo "To run specific variants, edit VARIANTS variable in script."
    exit 0
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
fi

# ============================================================
# Pre-flight checks
# ============================================================

echo "============================================================"
echo "🚀 Variant Batch Evaluation Script"
echo "============================================================"
echo ""
echo "📋 Configuration:"
echo "   Connection Info: ${CONNECTION_INFO}"
echo "   Episodes/Task:   ${NUM_EPISODES}"
echo "   Max Steps:       ${MAX_STEPS}"
echo "   Seed:            ${SEED}"
echo "   Output Dir:      ${OUTPUT_DIR}"
echo "   Domain ID:       ${DOMAIN_ID}"
echo "   Variants:        ${VARIANTS}"
echo ""

# Check if connection info exists
if [[ ! -f "${CONNECTION_INFO}" ]]; then
    echo "❌ Error: Connection info file not found: ${CONNECTION_INFO}"
    echo "   Please start the GTA-VLA server first and provide the correct path."
    exit 1
fi

echo "✅ Connection info found."
echo ""

# ============================================================
# Run evaluation
# ============================================================

echo "🔄 Starting variant batch evaluation..."
echo ""

uv run -m evaluation.simpler.WidowX.client_variant_batch \
    --variants ${VARIANTS} \
    --connection_info "${CONNECTION_INFO}" \
    --num_episodes ${NUM_EPISODES} \
    --max_steps ${MAX_STEPS} \
    --seed ${SEED} \
    --output_dir "${OUTPUT_DIR}" \
    --domain_id ${DOMAIN_ID}

echo ""
echo "============================================================"
echo "✅ Variant batch evaluation completed!"
echo "   Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
