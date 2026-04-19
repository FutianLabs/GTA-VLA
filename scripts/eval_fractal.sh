#!/bin/bash
set -euo pipefail
NO_SAVE_VIDEO=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CKPT_ROOT="${CKPT_ROOT:-$PROJECT_DIR/ckpt/fractal}"
SAVE_ROOT="${SAVE_ROOT:-$PROJECT_DIR/visualization/fractal_0311}"
SIMPLER_DIR="${SIMPLER_DIR:-/VLA-Data/scripts/lingyiran/SimplerEnv}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

EPISODES_PER_SCENARIO="${EPISODES_PER_SCENARIO:-1}"
OPEN_CLOSE_MAX_SCENARIOS="${OPEN_CLOSE_MAX_SCENARIOS:-8}"
if [[ -z "${TASKS+x}" ]]; then
  TASKS=("coke_can" "move_near" "open_close" "place_in")
fi
SETTINGS=("vm" "va")

SAVE_VIDEO_FLAG="--save_video"
if [[ "${NO_SAVE_VIDEO:-0}" == "1" ]]; then
  SAVE_VIDEO_FLAG="--no_save_video"
fi

mkdir -p "$SAVE_ROOT"

echo "CKPT_ROOT=$CKPT_ROOT"
echo "SAVE_ROOT=$SAVE_ROOT"
echo "SIMPLER_DIR=$SIMPLER_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "EPISODES_PER_SCENARIO=$EPISODES_PER_SCENARIO"
echo "OPEN_CLOSE_MAX_SCENARIOS=$OPEN_CLOSE_MAX_SCENARIOS"
echo "TASKS=${TASKS[*]}"
echo "SETTINGS=${SETTINGS[*]}"
echo "VIDEO_FLAG=$SAVE_VIDEO_FLAG"

declare -a CKPT_DIRS=()
if [[ -d "$CKPT_ROOT" && -f "$CKPT_ROOT/config.json" ]]; then
  CKPT_DIRS+=("$CKPT_ROOT")
elif [[ -d "$CKPT_ROOT" ]]; then
  shopt -s nullglob
  for d in "$CKPT_ROOT"/ckpt-*; do
    [[ -d "$d" && -f "$d/config.json" ]] && CKPT_DIRS+=("$d")
  done
  if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
    for exp_dir in "$CKPT_ROOT"/*; do
      [[ -d "$exp_dir" ]] || continue
      for d in "$exp_dir"/ckpt-*; do
        [[ -d "$d" && -f "$d/config.json" ]] && CKPT_DIRS+=("$d")
      done
    done
  fi
  shopt -u nullglob
fi

if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
  echo "No valid ckpt dirs found under: $CKPT_ROOT"
  exit 1
fi

for ckpt_dir in "${CKPT_DIRS[@]}"; do
    parent_name="$(basename "$(dirname "$ckpt_dir")")"
    ckpt_name="$(basename "$ckpt_dir")"
    if [[ "$parent_name" == "fractal" ]]; then
      exp_name="single_ckpt"
    else
      exp_name="$parent_name"
    fi

    run_dir="$SAVE_ROOT/$exp_name/$ckpt_name"
    out_dir="$run_dir/outputs"
    vis_dir="$run_dir/visualization"
    mkdir -p "$out_dir" "$vis_dir"

    echo "=================================================="
    echo "Running: $exp_name/$ckpt_name"
    echo "Model: $ckpt_dir"
    echo "=================================================="

    for setting in "${SETTINGS[@]}"; do
      echo "[${exp_name}/${ckpt_name}] setting=${setting}"
      PYTHONPATH="$PROJECT_DIR" SIMPLER_DIR="$SIMPLER_DIR" \
      "$PYTHON_BIN" "$PROJECT_DIR/debug_google_eval/evaluate_google_sim_debug.py" \
        --model_path "$ckpt_dir" \
        --google_setting "$setting" \
        --tasks "${TASKS[@]}" \
        --max_episodes_per_scenario "$EPISODES_PER_SCENARIO" \
        --max_scenarios_open_close "$OPEN_CLOSE_MAX_SCENARIOS" \
        $SAVE_VIDEO_FLAG \
        --output_root "$out_dir" \
        --visualization_root "$vis_dir"
    done

    python3 - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
out_dir = run_dir / "outputs"

report = {
    "run_dir": str(run_dir),
    "per_setting_task_success_rate": {},
    "setting_success_rate": {},
    "overall_success_rate": 0.0,
    "overall_episodes": 0,
    "overall_successes": 0,
}

all_done = []

for setting in ["vm", "va"]:
    s_dir = out_dir / setting
    task_map = {}
    setting_done = []

    if s_dir.exists():
        for task_dir in sorted([p for p in s_dir.iterdir() if p.is_dir()]):
            f = task_dir / "debug_results.jsonl"
            rows = []
            if f.exists():
                with f.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))

            dones = [1 if bool(r.get("done", False)) else 0 for r in rows]
            succ = sum(dones)
            eps = len(dones)
            rate = (succ / eps) if eps else 0.0
            task_map[task_dir.name] = {
                "success_rate": rate,
                "episodes": eps,
                "successes": succ,
            }
            setting_done.extend(dones)

    s_succ = sum(setting_done)
    s_eps = len(setting_done)
    report["per_setting_task_success_rate"][setting] = task_map
    report["setting_success_rate"][setting] = {
        "success_rate": (s_succ / s_eps) if s_eps else 0.0,
        "episodes": s_eps,
        "successes": s_succ,
    }
    all_done.extend(setting_done)

report["overall_episodes"] = len(all_done)
report["overall_successes"] = sum(all_done)
report["overall_success_rate"] = (sum(all_done) / len(all_done)) if all_done else 0.0

with (run_dir / "success_report.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps({
    "ckpt": str(run_dir),
    "overall_success_rate": report["overall_success_rate"],
    "overall_episodes": report["overall_episodes"],
}, ensure_ascii=False))
PY
done

echo "All done. Reports are in: $SAVE_ROOT"
