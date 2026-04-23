import argparse
import os
import json
import h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

_roots_env = os.getenv("GTA_VLA_ROBOMIND_ROOTS", "")
if _roots_env.strip():
    DATA_ROOTS = [p for p in _roots_env.split(":") if p.strip()]
else:
    DATA_ROOTS = [os.path.expanduser("~/data/x-humanoid-robomind/RoboMIND/benchmark1_0_extracted")]

SINGLE_ARM_GROUPS = {
    "robomind-franka": {
        "folders": ["h5_franka_1rgb", "h5_franka_3rgb"],
        "observation_key": ["observations/rgb_images/camera_left", "observations/rgb_images/camera_right", "observations/rgb_images/camera_top"],
        "check_keys": ["puppet/end_effector", "puppet/joint_position"],
    },
    "robomind-ur": {
        "folders": ["h5_ur_1rgb"],
        "observation_key": ["observations/rgb_images/camera_top"],
        "check_keys": ["puppet/end_effector", "puppet/joint_position"],
    },
    "robomind-sim-franka": {
        "folders": ["h5_simulation", "h5_sim_franka_3rgb"],
        "observation_key": ["observations/rgb_images/camera_front_external", "observations/rgb_images/camera_handeye",
                            "observations/rgb_images/camera_left_external", "observations/rgb_images/camera_right_external"],
        "check_keys": ["franka/end_effector", "franka/joint_position"],
    },
}


def collect_paths_via_find(data_roots, folder_names):
    """Use shell find to collect trajectory.hdf5 paths (much faster on NFS)."""
    import subprocess
    paths = []
    for root in data_roots:
        for folder in folder_names:
            d = os.path.join(root, folder)
            if not os.path.isdir(d):
                continue
            result = subprocess.run(
                ["find", d, "-name", "trajectory.hdf5", "-type", "f"],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                paths.extend(line for line in result.stdout.strip().split("\n") if line)
    return paths


def load_paths_from_file(path_file):
    with open(path_file) as f:
        return [line.strip() for line in f if line.strip()]


def check_h5_valid(args_tuple):
    filepath, check_keys = args_tuple
    try:
        with h5py.File(filepath, "r") as f:
            if "language_raw" not in f:
                return None
            lang = f["language_raw"][0]
            if isinstance(lang, bytes):
                lang = lang.decode()
            if not lang.strip():
                return None
            for key in check_keys:
                if key not in f:
                    return None
        return filepath
    except Exception:
        return None


def generate_meta(output_dir, num_workers=16, filter_empty=True, path_files=None):
    os.makedirs(output_dir, exist_ok=True)

    for group_name, cfg in SINGLE_ARM_GROUPS.items():
        if path_files and group_name in path_files:
            print(f"\n[{group_name}] Loading paths from {path_files[group_name]}...")
            all_h5 = load_paths_from_file(path_files[group_name])
        else:
            print(f"\n[{group_name}] Collecting paths via find from {cfg['folders']}...")
            all_h5 = collect_paths_via_find(DATA_ROOTS, cfg["folders"])

        print(f"[{group_name}] Found {len(all_h5)} trajectory files.")

        if not all_h5:
            print(f"[{group_name}] Skipping.")
            continue

        if filter_empty:
            print(f"[{group_name}] Validating with {num_workers} workers...")
            work_items = [(fp, cfg["check_keys"]) for fp in all_h5]
            datalist = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for result in tqdm(executor.map(check_h5_valid, work_items, chunksize=64),
                                   total=len(work_items), desc=group_name):
                    if result is not None:
                        datalist.append(result)
        else:
            datalist = all_h5

        datalist = sorted(datalist)
        print(f"[{group_name}] Valid: {len(datalist)} / {len(all_h5)}")

        meta = {
            "dataset_name": group_name,
            "observation_key": cfg["observation_key"],
            "language_instruction_key": "language_raw",
            "datalist": datalist,
        }

        out_path = os.path.join(output_dir, f"{group_name}_meta.json")
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[{group_name}] Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data"))
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--no_filter", action="store_true",
                        help="Skip hdf5 validation, include all trajectory files")
    parser.add_argument("--path_files", type=str, nargs="*",
                        help="Pre-collected path files: group_name=filepath pairs, e.g. robomind-franka=/tmp/franka.txt")
    args = parser.parse_args()

    path_files = {}
    if args.path_files:
        for pf in args.path_files:
            name, fpath = pf.split("=", 1)
            path_files[name] = fpath

    generate_meta(args.output_dir, args.num_workers,
                  filter_empty=not args.no_filter, path_files=path_files)


if __name__ == "__main__":
    main()
