"""
Generate Bridge-style meta JSON for training.

Scans an output directory of episode_*.hdf5 files and produces a meta file
compatible with BridgeHandler / BridgeCotHandler data loading.

Usage:
    python -m datasets.tools.sim_data_builder.gen_meta \
        --data_dir output/bridge_widowx_sim \
        --output_path output/bridge_widowx_sim_meta.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py


def generate_meta(
    data_dir: str,
    output_path: str,
    dataset_name: str = "Bridge",
    include_wrist: bool = True,
) -> dict:
    """Scan data_dir for episode HDF5 files and build meta JSON.

    Args:
        data_dir: Directory containing episode_*.hdf5 files.
        output_path: Where to write the meta JSON.
        dataset_name: "Bridge" for BridgeHandler.
        include_wrist: Whether to include image_3 in observation_key
                       for episodes with valid wrist.

    Returns:
        The meta dict that was written.
    """
    data_dir = Path(data_dir)
    episodes = sorted(data_dir.glob("episode_*.hdf5"))

    if not episodes:
        print(f"No episode_*.hdf5 found in {data_dir}")
        sys.exit(1)

    root_abs = os.path.abspath(str(data_dir))
    datalist = [os.path.join(root_abs, os.path.basename(ep)) for ep in episodes]

    # Determine observation keys from first file
    obs_keys = ["observation/image_0"]
    optional_keys = ["observation/image_1", "observation/image_2"]

    # Check if any wrist views are valid
    has_any_wrist = False
    if include_wrist:
        for ep_path in episodes:
            with h5py.File(str(ep_path), "r") as f:
                if f.attrs.get("wrist_view_valid", False):
                    has_any_wrist = True
                    break

    if has_any_wrist:
        obs_keys.append("observation/image_3")
    else:
        optional_keys.append("observation/image_3")

    meta = {
        "observation_key": obs_keys,
        "optional_view_key": optional_keys,
        "dataset_name": dataset_name,
        "language_instruction_key": "instruction",
        "datalist": datalist,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Meta written to {output_path}")
    print(f"  Episodes: {len(datalist)}")
    print(f"  Observation keys: {obs_keys}")
    print(f"  Wrist valid in any episode: {has_any_wrist}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Generate Bridge meta JSON")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with episode_*.hdf5 files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output meta JSON path")
    parser.add_argument("--dataset_name", type=str, default="Bridge")
    parser.add_argument("--no_wrist", action="store_true",
                        help="Exclude image_3 from observation_key")
    args = parser.parse_args()

    generate_meta(
        data_dir=args.data_dir,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        include_wrist=not args.no_wrist,
    )


if __name__ == "__main__":
    main()
