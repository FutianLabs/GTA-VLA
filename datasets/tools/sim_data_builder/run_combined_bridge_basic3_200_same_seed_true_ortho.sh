#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/VLA-Data/scripts/lingyiran/x-vla-main"
OUTPUT_DIR="${1:-/VLA-Data/scripts/lianqing/data/openX/x-vla/bridge_objects_basic3_200_combined_same_seed_true_ortho}"
TARGET_SUCCESS="${TARGET_SUCCESS:-200}"
MAX_STEPS="${MAX_STEPS:-120}"
MAX_TOTAL_EPISODE_IDS="${MAX_TOTAL_EPISODE_IDS:-20000}"
RESUME="${RESUME:-0}"
SKIP_LABELS="${SKIP_LABELS:-}"
DROP_LABELS="${DROP_LABELS:-}"
RESTART_EVERY_SUCCESSES="${RESTART_EVERY_SUCCESSES:-25}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1200}"

if [ "$RESUME" != "1" ] && [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
  echo "Output directory is not empty: $OUTPUT_DIR" >&2
  echo "Please pass a fresh directory path." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1
export OUTPUT_DIR
export TARGET_SUCCESS
export MAX_STEPS
export MAX_TOTAL_EPISODE_IDS
export RESUME
export SKIP_LABELS
export DROP_LABELS
export RESTART_EVERY_SUCCESSES
export CHUNK_TIMEOUT_SECONDS

cd "$ROOT_DIR"

CASE_LABELS=(
  bridge_objects_base
  bridge_objects_lighting
  bridge_objects_distractor
  bridge_objects_camera
  bridge_objects_table_color
  widowx_carrot_on_plate
  widowx_spoon_on_towel
  widowx_stack_cube
)

should_skip_label() {
  local label="$1"
  [[ ",${SKIP_LABELS}," == *",${label},"* ]]
}

for CASE_LABEL in "${CASE_LABELS[@]}"; do
  if should_skip_label "$CASE_LABEL"; then
    echo "SKIP label=$CASE_LABEL"
    continue
  fi

  export CASE_LABEL
    while true; do
    set +e
    timeout --signal=TERM --kill-after=30s "${CHUNK_TIMEOUT_SECONDS}" python3 - <<'PY'
from pathlib import Path
import json
import os
import sys
import time

sys.path.insert(0, "/VLA-Data/scripts/lingyiran/x-vla-main")

from datasets.tools.sim_data_builder.collector import collect_episode
from datasets.tools.sim_data_builder.run import _write_episode, _write_manifest_entry
from datasets.tools.sim_data_builder.waypoint_policy import WaypointPutOnPolicy

OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
TARGET_SUCCESS = int(os.environ["TARGET_SUCCESS"])
MAX_STEPS = int(os.environ["MAX_STEPS"])
MAX_TOTAL_EPISODE_IDS = int(os.environ["MAX_TOTAL_EPISODE_IDS"])
RESUME = os.environ.get("RESUME", "0") == "1"
DROP_LABELS = {x.strip() for x in os.environ.get("DROP_LABELS", "").split(",") if x.strip()}
CASE_LABEL = os.environ["CASE_LABEL"]
RESTART_EVERY_SUCCESSES = int(os.environ.get("RESTART_EVERY_SUCCESSES", "25"))

CASES = [
    {
        "label": "bridge_objects_base",
        "task_key": "widowx_put_bridge_objects_on_plate",
        "env_modifiers": None,
        "modifier_params": None,
    },
    {
        "label": "bridge_objects_lighting",
        "task_key": "widowx_put_bridge_objects_on_plate",
        "env_modifiers": ["lighting"],
        "modifier_params": None,
    },
    {
        "label": "bridge_objects_distractor",
        "task_key": "widowx_put_bridge_objects_on_plate",
        "env_modifiers": ["distractor"],
        "modifier_params": None,
    },
    {
        "label": "bridge_objects_camera",
        "task_key": "widowx_put_bridge_objects_on_plate",
        "env_modifiers": ["camera"],
        "modifier_params": None,
    },
    {
        "label": "bridge_objects_table_color",
        "task_key": "widowx_put_bridge_objects_on_plate",
        "env_modifiers": ["table_color"],
        "modifier_params": None,
    },
    {
        "label": "widowx_carrot_on_plate",
        "task_key": "widowx_carrot_on_plate",
        "env_modifiers": None,
        "modifier_params": None,
    },
    {
        "label": "widowx_spoon_on_towel",
        "task_key": "widowx_spoon_on_towel",
        "env_modifiers": None,
        "modifier_params": None,
    },
    {
        "label": "widowx_stack_cube",
        "task_key": "widowx_stack_cube",
        "env_modifiers": None,
        "modifier_params": None,
    },
]

case_by_label = {case["label"]: case for case in CASES}
case = case_by_label[CASE_LABEL]


def write_manifest(manifest):
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ordered_summary(summary_by_label):
    return [summary_by_label[c["label"]] for c in CASES if c["label"] in summary_by_label]


def write_summary(summary_by_label):
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(ordered_summary(summary_by_label), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rebuild_partial_progress(manifest, summary_by_label):
    manifest_by_label = {}
    for entry in manifest:
        label = entry.get("variant_label")
        if not label:
            continue
        manifest_by_label.setdefault(label, []).append(entry)

    for item in CASES:
        label = item["label"]
        entries = manifest_by_label.get(label, [])
        prev = summary_by_label.get(label, {})
        manifest_successes = len(entries)
        if not entries and not prev:
            continue
        consumed = int(prev.get("episode_ids_consumed", 0))
        if entries:
            consumed = max(consumed, max(int(e.get("env_episode_id", -1)) for e in entries) + 1)
        successes = max(int(prev.get("successes", 0)), manifest_successes)
        summary_by_label[label] = {
            "label": label,
            "task_key": item["task_key"],
            "env_modifiers": item["env_modifiers"] or [],
            "modifier_params": item["modifier_params"] or {},
            "successes": successes,
            "target_successes": TARGET_SUCCESS,
            "episode_ids_consumed": consumed,
            "elapsed_sec": float(prev.get("elapsed_sec", 0.0)),
        }


manifest_path = OUTPUT_DIR / "manifest.json"
summary_path = OUTPUT_DIR / "summary.json"

manifest = []
summary = []
if RESUME and manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if RESUME and summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

if DROP_LABELS:
    kept_manifest = []
    for entry in manifest:
        if entry.get("variant_label") in DROP_LABELS:
            h5_path = entry.get("h5_path")
            if h5_path:
                try:
                    Path(h5_path).unlink()
                except FileNotFoundError:
                    pass
            continue
        kept_manifest.append(entry)
    manifest = kept_manifest
    write_manifest(manifest)

summary_by_label = {
    row["label"]: row for row in summary if row.get("label") not in DROP_LABELS
}
rebuild_partial_progress(manifest, summary_by_label)
write_summary(summary_by_label)

task_key = case["task_key"]
env_modifiers = case["env_modifiers"]
modifier_params = case["modifier_params"]
prev = summary_by_label.get(CASE_LABEL, {})
success_count = int(prev.get("successes", 0))
initial_success_count = success_count
episode_id = int(prev.get("episode_ids_consumed", 0))
elapsed_sec = float(prev.get("elapsed_sec", 0.0))
segment_start = time.time()
global_idx = len(manifest)

if success_count >= TARGET_SUCCESS:
    print(f"ALREADY_DONE label={CASE_LABEL} successes={success_count}/{TARGET_SUCCESS}", flush=True)
    raise SystemExit(0)

print(
    f"\n=== CASE {CASE_LABEL} task={task_key} need={TARGET_SUCCESS} modifiers={env_modifiers or []} ===",
    flush=True,
)

while success_count < TARGET_SUCCESS and episode_id < MAX_TOTAL_EPISODE_IDS:
    t0 = time.time()
    try:
        policy = WaypointPutOnPolicy(task_key)
        data = collect_episode(
            task_key=task_key,
            episode_id=episode_id,
            policy=policy,
            max_steps=MAX_STEPS,
            env_modifiers=env_modifiers,
            modifier_params=modifier_params,
        )
    except Exception as e:
        print(f"  ERROR label={CASE_LABEL} env_ep={episode_id}: {e}", flush=True)
        episode_id += 1
        elapsed_sec += time.time() - segment_start
        summary_by_label[CASE_LABEL] = {
            "label": CASE_LABEL,
            "task_key": task_key,
            "env_modifiers": env_modifiers or [],
            "modifier_params": modifier_params or {},
            "successes": success_count,
            "target_successes": TARGET_SUCCESS,
            "episode_ids_consumed": episode_id,
            "elapsed_sec": elapsed_sec,
        }
        write_summary(summary_by_label)
        segment_start = time.time()
        continue

    if data.success:
        h5_path = _write_episode(data, OUTPUT_DIR, global_idx)
        entry = _write_manifest_entry(
            data,
            global_idx,
            h5_path,
            variant_label=CASE_LABEL,
            env_modifiers=env_modifiers,
            modifier_params=modifier_params,
        )
        manifest.append(entry)
        global_idx += 1
        success_count += 1
        write_manifest(manifest)

        summary_by_label[CASE_LABEL] = {
            "label": CASE_LABEL,
            "task_key": task_key,
            "env_modifiers": env_modifiers or [],
            "modifier_params": modifier_params or {},
            "successes": success_count,
            "target_successes": TARGET_SUCCESS,
            "episode_ids_consumed": episode_id + 1,
            "elapsed_sec": elapsed_sec + (time.time() - segment_start),
        }
        write_summary(summary_by_label)
        elapsed_sec += time.time() - segment_start
        segment_start = time.time()

        if success_count % 10 == 0 or success_count == 1:
            print(
                f"  [{global_idx - 1:06d}] success={success_count:03d}/{TARGET_SUCCESS:03d} "
                f"label={CASE_LABEL} env_ep={episode_id:05d} src={data.source_object_name!r} "
                f"dt={time.time() - t0:.1f}s",
                flush=True,
            )

    episode_id += 1

    if RESTART_EVERY_SUCCESSES > 0 and (success_count - initial_success_count) >= RESTART_EVERY_SUCCESSES:
        break

elapsed_sec += time.time() - segment_start
summary_by_label[CASE_LABEL] = {
    "label": CASE_LABEL,
    "task_key": task_key,
    "env_modifiers": env_modifiers or [],
    "modifier_params": modifier_params or {},
    "successes": success_count,
    "target_successes": TARGET_SUCCESS,
    "episode_ids_consumed": episode_id,
    "elapsed_sec": elapsed_sec,
}
write_manifest(manifest)
write_summary(summary_by_label)

if success_count < TARGET_SUCCESS and episode_id < MAX_TOTAL_EPISODE_IDS and (success_count - initial_success_count) >= RESTART_EVERY_SUCCESSES:
        print(
                f"CASE_CHUNK_DONE label={CASE_LABEL} successes={success_count}/{TARGET_SUCCESS} consumed={episode_id}",
                flush=True,
        )
        raise SystemExit(42)

print(
    f"CASE_DONE label={CASE_LABEL} successes={success_count}/{TARGET_SUCCESS} consumed={episode_id}",
    flush=True,
)
PY
    status=$?
    set -e
    if [ "$status" -eq 0 ]; then
        break
    fi
    if [ "$status" -eq 42 ]; then
        echo "RESTART_CASE label=$CASE_LABEL"
        continue
    fi
    if [ "$status" -eq 124 ] || [ "$status" -eq 137 ] || [ "$status" -eq 143 ]; then
        echo "RESTART_CASE_TIMEOUT label=$CASE_LABEL timeout=${CHUNK_TIMEOUT_SECONDS}s"
        continue
    fi
    exit "$status"
    done
done

python3 - <<'PY'
from pathlib import Path
import json
import os
import sys

sys.path.insert(0, "/VLA-Data/scripts/lingyiran/x-vla-main")

from datasets.tools.sim_data_builder.gen_meta import generate_meta

output_dir = Path(os.environ["OUTPUT_DIR"])
manifest_path = output_dir / "manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest:
        generate_meta(data_dir=str(output_dir), output_path=str(output_dir / "meta.json"))

print(f"\nDONE output_dir={output_dir}", flush=True)
PY
