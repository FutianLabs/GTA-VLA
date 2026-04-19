import argparse
import glob
import os
import json
import mmengine
import h5py
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from mmengine import fileio
from tqdm import tqdm


def _open_h5(path: str) -> h5py.File:
    try:
        return h5py.File(path, "r")
    except OSError:
        return h5py.File(io.BytesIO(fileio.get(path)), "r")


def read_instruction_from_h5(filepath: str, instruction_key="instruction") -> str:
    try:
        keys = instruction_key if isinstance(instruction_key, list) else [instruction_key]
        with _open_h5(filepath) as f:
            for key in keys:
                try:
                    if key in f.attrs:
                        v = f.attrs[key]
                        lang = v.decode() if isinstance(v, bytes) else v
                    elif key in f:
                        ds = f[key]
                        v = ds[()]
                        lang = v.decode() if getattr(ds, "shape", ()) == () else v[0].decode()
                    else:
                        continue
                    if isinstance(lang, list):
                        lang = lang[0] if lang else ""
                    if isinstance(lang, bytes):
                        lang = lang.decode("utf-8")
                    if lang.strip():
                        return lang.strip()
                except Exception:
                    continue
        return ""
    except Exception as e:
        print(f"Warning: Could not read instruction from {filepath}: {e}")
        return ""


def check_file_has_instruction(filepath: str, instruction_key: str = "instruction") -> tuple:
    instruction = read_instruction_from_h5(filepath, instruction_key)
    return (filepath, bool(instruction))


def _check_agibot_episode(args_tuple):
    task_id, ep_id, proprio_root, obs_root = args_tuple
    proprio_path = os.path.join(proprio_root, task_id, ep_id, "proprio_stats.h5")
    obs_dir = os.path.join(obs_root, task_id, ep_id, "videos")
    if not os.path.isfile(proprio_path):
        return None
    if not os.path.isdir(obs_dir):
        return None
    required_cams = ["head_color.mp4", "hand_left_color.mp4", "hand_right_color.mp4"]
    for cam in required_cams:
        if not os.path.isfile(os.path.join(obs_dir, cam)):
            return None
    try:
        with h5py.File(proprio_path, "r") as f:
            T = f["state"]["joint"]["position"].shape[0]
            if T < 30:
                return None
    except Exception:
        return None
    return f"{task_id}/{ep_id}"


def generate_agibot_meta(num_workers: int = 32):
    data_root = "/VLA-Data/scripts/lingyiran/data/agibot-world/AgiBotWorld-Alpha"
    proprio_root = os.path.join(data_root, "proprio_stats")
    obs_root = os.path.join(data_root, "observations")
    annotation_dir = "/VLA-Data/scripts/lingyiran/data/agibot-world/annotations"

    tasks = sorted(os.listdir(proprio_root))
    work_items = []
    for task_id in tasks:
        task_dir = os.path.join(proprio_root, task_id)
        if not os.path.isdir(task_dir):
            continue
        for ep_id in os.listdir(task_dir):
            work_items.append((task_id, ep_id, proprio_root, obs_root))

    print(f"Scanning {len(work_items)} candidate episodes with {num_workers} workers...")

    datalist = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(_check_agibot_episode, work_items), total=len(work_items)):
            if result is not None:
                datalist.append(result)

    datalist = sorted(datalist)
    print(f"Valid episodes: {len(datalist)}")

    meta = {
        "dataset_name": "agiworld-cot",
        "sub_dataset_name": "agiworld-on-site-pack-extra",
        "data_root": data_root,
        "annotation_dir": annotation_dir,
        "datalist": datalist,
        "cot_config": {
            "coord_scale": 1000,
            "gripper_future_steps": 5,
            "detector_priority": ["seed_vl", "dino_x"]
        }
    }

    output_path = os.path.join(os.path.dirname(__file__), "../../data/agibot_meta_cot.json")
    output_path = os.path.abspath(output_path)
    mmengine.dump(meta, output_path)
    print(f"Saved to: {output_path}")


def _check_robomind_h5(filepath):
    try:
        with h5py.File(filepath, "r") as f:
            if "puppet" not in f:
                return None
            if "observations/rgb_images/camera_front" not in f:
                return None
            T = f["puppet"]["end_effector_left"].shape[0]
            if T < 30:
                return None
        return filepath
    except Exception:
        return None


def generate_robomind_meta(num_workers: int = 32):
    data_root = "/VLA-Data/scripts/lingyiran/data/x-humanoid-robomind/RoboMIND/benchmark1_0_extracted/h5_agilex_3rgb"
    annotation_dir = "/VLA-Data/scripts/lingyiran/data/x-humanoid-robomind/annotations"

    all_h5 = glob.glob(os.path.join(data_root, "**/trajectory.hdf5"), recursive=True)
    print(f"Found {len(all_h5)} trajectory files, validating with {num_workers} workers...")

    datalist = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(_check_robomind_h5, all_h5), total=len(all_h5)):
            if result is not None:
                datalist.append(result)

    datalist = sorted(datalist)
    print(f"Valid trajectories: {len(datalist)}")

    meta = {
        "dataset_name": "robomind-cot",
        "sub_dataset_name": "robomind-agilex",
        "observation_key": ["observations/rgb_images/camera_front"],
        "language_instruction_key": "language_raw",
        "annotation_dir": annotation_dir,
        "datalist": datalist,
        "cot_config": {
            "coord_scale": 1000,
            "gripper_future_steps": 5,
            "detector_priority": ["seed_vl", "dino_x"]
        }
    }

    output_path = os.path.join(os.path.dirname(__file__), "../../data/robomind_meta_cot.json")
    output_path = os.path.abspath(output_path)
    mmengine.dump(meta, output_path)
    print(f"Saved to: {output_path}")


def generate_libero_meta():
    meta = dict()
    meta["observation_key"] = ["observation/agentview_rgb", "observation/eye_in_hand_rgb"]
    meta["dataset_name"] = "libero"
    meta["language_instruction_key"] = "instruction"
    filelist = glob.glob("/VLA-Data/scripts/lianqing/data/libero_xvla_flip/*no_noops/*hdf5")
    filelist = [i for i in filelist if "libero_90_no_noops" not in i]
    filelist = sorted(filelist)
    meta['datalist'] = filelist
    print(f"Total files: {len(filelist)}")
    output_path = "/VLA-Data/scripts/lianqing/projects/vla/X-VLA/data/libero_meta.json"
    mmengine.dump(meta, output_path)
    print(f"Saved to: {output_path}")


def generate_libero_mix_meta():
    meta = dict()
    meta["observation_key"] = ["observation/agentview_rgb", "observation/eye_in_hand_rgb"]
    meta["dataset_name"] = "libero"
    meta["language_instruction_key"] = "instruction"
    filelist = glob.glob("/VLA-Data/scripts/lianqing/data/libero_plus_training_30k/*hdf5")
    filelist = sorted(filelist)
    meta['datalist'] = filelist
    print(f"Total files: {len(filelist)}")
    output_path = "/VLA-Data/scripts/lianqing/projects/vla/X-VLA/data/libero_mix_meta.json"
    mmengine.dump(meta, output_path)
    print(f"Saved to: {output_path}")


def generate_bridge_meta(num_workers: int = 32):
    meta = dict()
    meta["observation_key"] = ["observation/image_0"]
    meta["dataset_name"] = "Bridge"
    meta["language_instruction_key"] = "instruction"
    filelist = glob.glob("/VLA-Data/scripts/lianqing/data/openX/x-vla/bridge/*hdf5")
    filelist = sorted(filelist)
    print(f"Total files found: {len(filelist)}")
    filtered_filelist = []
    empty_count = 0
    check_func = partial(check_file_has_instruction, instruction_key=meta["language_instruction_key"])
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_func, filepath): filepath for filepath in filelist}
        for future in tqdm(as_completed(futures), total=len(filelist), desc="Processing"):
            try:
                filepath, has_instruction = future.result()
                if has_instruction:
                    filtered_filelist.append(filepath)
                else:
                    empty_count += 1
            except Exception as e:
                print(f"\nError: {e}")
                empty_count += 1
    filtered_filelist = sorted(filtered_filelist)
    meta['datalist'] = filtered_filelist
    print(f"Empty: {empty_count}, Valid: {len(filtered_filelist)}")
    output_path = "/VLA-Data/scripts/lianqing/projects/vla/X-VLA/data/bridge_meta_ins.json"
    mmengine.dump(meta, output_path)
    print(f"Saved to: {output_path}")


def _check_droid_h5(args):
    """Return filepath if it matches the requested calib_mode and has instruction.
    
    Args:
        args: (filepath, calib_mode) where calib_mode is 'direct' or 'non_direct'.
              'non_direct' matches propagated or no calibration (anything != 'direct').
    """
    filepath, calib_mode = args
    try:
        with h5py.File(filepath, "r") as f:
            if "gripper_position" not in f:
                return None
            source = f.attrs.get("calib_source", "")
            if isinstance(source, bytes):
                source = source.decode("utf-8")
            if calib_mode == "direct" and source != "direct":
                return None
            if calib_mode == "non_direct" and source == "direct":
                return None
            lang = f.attrs.get("language_instruction", b"")
            if isinstance(lang, bytes):
                lang = lang.decode("utf-8")
            if not lang.strip():
                return None
        return filepath
    except Exception:
        return None


def generate_droid_meta(num_workers: int = 32, calib_mode: str = "direct"):
    data_root = "/VLA-Data/scripts/lianqing/data/openX/x-vla/droid"
    meta = {
        "observation_key": [
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
        ],
        "dataset_name": "DroidCot-Left",
        "language_instruction_key": [
            "language_instruction",
            "language_instruction_1",
            "language_instruction_2",
            "language_instruction_3",
        ],
        "dataset_config": {
            "image_keys": [
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
            ],
            "instruction_key": [
                "language_instruction",
                "language_instruction_1",
                "language_instruction_2",
                "language_instruction_3",
            ],
        },
    }

    filelist = sorted(glob.glob(os.path.join(data_root, "*.hdf5")))
    print(f"Total files found: {len(filelist)}")
    print(f"Calib mode: {calib_mode}")

    work_items = [(fp, calib_mode) for fp in filelist]
    filtered = []
    skipped = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(_check_droid_h5, work_items),
                           total=len(filelist), desc=f"Filtering {calib_mode}-calib"):
            if result is not None:
                filtered.append(result)
            else:
                skipped += 1

    filtered = sorted(filtered)
    meta["datalist"] = filtered
    print(f"{calib_mode}-calib with instruction: {len(filtered)}, Skipped: {skipped}")

    suffix = f"_{calib_mode}"
    output_path = os.path.join(os.path.dirname(__file__), f"../../data/droid_meta{suffix}.json")
    output_path = os.path.abspath(output_path)
    mmengine.dump(meta, output_path)
    print(f"Saved to: {output_path}")

    import random
    dummy_meta = dict(meta)
    random.seed(42)
    dummy_meta["datalist"] = sorted(random.sample(filtered, min(100, len(filtered))))
    dummy_path = os.path.join(os.path.dirname(__file__), f"../../data/droid_meta_dummy{suffix}.json")
    dummy_path = os.path.abspath(dummy_path)
    mmengine.dump(dummy_meta, dummy_path)
    print(f"Saved dummy ({len(dummy_meta['datalist'])} episodes) to: {dummy_path}")


def generate_fractal_meta(num_workers: int = 32):
    data_root = os.path.expanduser("~/data/openX/x-vla/fractal")
    meta = dict()
    meta["observation_key"] = ["observation/image_0"]
    meta["dataset_name"] = "Fractal"
    meta["language_instruction_key"] = "instruction"

    filelist = glob.glob(os.path.join(data_root, "*.hdf5"))
    filelist = sorted(filelist)
    print(f"Total files found: {len(filelist)}")

    filtered_filelist = []
    empty_count = 0
    check_func = partial(check_file_has_instruction, instruction_key=meta["language_instruction_key"])
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_func, filepath): filepath for filepath in filelist}
        for future in tqdm(as_completed(futures), total=len(filelist), desc="Processing"):
            try:
                filepath, has_instruction = future.result()
                if has_instruction:
                    filtered_filelist.append(filepath)
                else:
                    empty_count += 1
            except Exception as e:
                print(f"\nError: {e}")
                empty_count += 1

    filtered_filelist = sorted(filtered_filelist)
    meta["datalist"] = filtered_filelist
    print(f"Empty: {empty_count}, Valid: {len(filtered_filelist)}")

    output_path = os.path.join(os.path.dirname(__file__), "../../data/fractal_meta.json")
    output_path = os.path.abspath(output_path)
    mmengine.dump(meta, output_path)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["libero", "bridge", "libero_mix", "agibot", "robomind", "fractal", "droid"])
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--calib", type=str, default="direct",
                        choices=["direct", "non_direct"],
                        help="Droid only: 'direct' for direct-calib, 'non_direct' for propagated/none")
    args = parser.parse_args()

    if args.dataset == "libero":
        generate_libero_meta()
    elif args.dataset == "bridge":
        generate_bridge_meta(num_workers=args.num_workers)
    elif args.dataset == "libero_mix":
        generate_libero_mix_meta()
    elif args.dataset == "agibot":
        generate_agibot_meta(num_workers=args.num_workers)
    elif args.dataset == "robomind":
        generate_robomind_meta(num_workers=args.num_workers)
    elif args.dataset == "fractal":
        generate_fractal_meta(num_workers=args.num_workers)
    elif args.dataset == "droid":
        generate_droid_meta(num_workers=args.num_workers, calib_mode=args.calib)


if __name__ == "__main__":
    main()
