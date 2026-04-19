"""
Smoke test: verify generated HDF5 files can be read by BridgeHandler.

Usage:
    python -m datasets.tools.sim_data_builder.smoke_test \
        --hdf5 output/episode_000000.hdf5

    python -m datasets.tools.sim_data_builder.smoke_test \
        --dir output/
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def verify_episode(path: str) -> bool:
    """Verify a single HDF5 file matches BridgeHandler expectations.

    Checks:
        1. /observation/image_0 exists, shape [T, H, W, 3], dtype uint8
        2. /proprio exists, shape [T, D] with D >= 6, dtype float32
        3. /action exists, last dim is gripper in [0, 1]
        4. attrs: instruction (str), instruction_source (str), wrist_view_valid (bool)
        5. If wrist_view_valid, /observation/image_3 exists with matching T
    """
    ok = True

    def _fail(msg):
        nonlocal ok
        print(f"  FAIL: {msg}")
        ok = False

    print(f"Checking {path} ...")
    try:
        with h5py.File(path, "r") as f:
            # ── image_0 ─────────────────────────────────────────────────
            if "observation/image_0" not in f:
                _fail("Missing /observation/image_0")
            else:
                img0 = f["observation/image_0"]
                if len(img0.shape) != 4 or img0.shape[-1] != 3:
                    _fail(f"image_0 shape unexpected: {img0.shape}")
                if img0.dtype != np.uint8:
                    _fail(f"image_0 dtype unexpected: {img0.dtype}")
                T = img0.shape[0]

            # ── proprio ─────────────────────────────────────────────────
            if "proprio" not in f:
                _fail("Missing /proprio")
            else:
                prop = f["proprio"]
                if len(prop.shape) != 2:
                    _fail(f"proprio should be 2-D, got {prop.shape}")
                elif prop.shape[1] < 6:
                    _fail(f"proprio dim < 6: {prop.shape}")
                if prop.dtype != np.float32:
                    _fail(f"proprio dtype unexpected: {prop.dtype}")
                if prop.shape[0] != T:
                    _fail(f"proprio T={prop.shape[0]} != image_0 T={T}")
                else:
                    # Check for NaN/Inf
                    data = prop[()]
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        _fail("proprio contains NaN or Inf")

            # ── action ──────────────────────────────────────────────────
            if "action" not in f:
                _fail("Missing /action")
            else:
                act = f["action"]
                if len(act.shape) != 2:
                    _fail(f"action should be 2-D, got {act.shape}")
                if act.dtype != np.float32:
                    _fail(f"action dtype unexpected: {act.dtype}")
                if act.shape[0] != T:
                    _fail(f"action T={act.shape[0]} != image_0 T={T}")
                else:
                    grip = act[()][:, -1]
                    if np.any(grip < -0.01) or np.any(grip > 1.01):
                        _fail(f"action gripper out of [0,1]: min={grip.min():.3f} max={grip.max():.3f}")

            # ── attrs ───────────────────────────────────────────────────
            for attr in ["instruction", "instruction_source", "wrist_view_valid"]:
                if attr not in f.attrs:
                    _fail(f"Missing attr '{attr}'")

            if "instruction" in f.attrs:
                instr = f.attrs["instruction"]
                if isinstance(instr, bytes):
                    instr = instr.decode()
                if not isinstance(instr, str) or len(instr.strip()) == 0:
                    _fail(f"instruction is empty or not a string")

            # ── wrist view ──────────────────────────────────────────────
            wrist_valid = f.attrs.get("wrist_view_valid", False)
            if wrist_valid:
                if "observation/image_3" not in f:
                    _fail("wrist_view_valid=True but /observation/image_3 missing")
                else:
                    img3 = f["observation/image_3"]
                    if img3.shape[0] != T:
                        _fail(f"image_3 T={img3.shape[0]} != image_0 T={T}")

            # ── Summary ─────────────────────────────────────────────────
            if ok:
                instr_str = f.attrs.get("instruction", "")
                if isinstance(instr_str, bytes):
                    instr_str = instr_str.decode()
                print(f"  OK  T={T}  proprio={f['proprio'].shape}  action={f['action'].shape}  "
                      f"wrist={'yes' if wrist_valid else 'no'}  "
                      f"instr=\"{instr_str[:60]}\"")

    except Exception as e:
        _fail(f"Exception reading file: {e}")

    return ok


def verify_with_bridge_handler(path: str) -> bool:
    """Attempt to actually load the file through BridgeHandler.build_left_right()."""
    try:
        from datasets.domain_handler.simulations import BridgeHandler
    except ImportError:
        print("  SKIP: BridgeHandler import failed (run from repo root)")
        return True

    print(f"  BridgeHandler test on {path} ...")
    try:
        with h5py.File(path, "r") as f:
            proprio = f["proprio"][()]
            action = f["action"][()]

        handler = BridgeHandler.__new__(BridgeHandler)
        handler.meta = {
            "observation_key": ["observation/image_0"],
            "dataset_name": "Bridge",
            "language_instruction_key": "instruction",
        }
        left, right, _, _, freq, qdur = handler.build_left_right(proprio, action)
        assert left.shape[-1] == 10, f"left dim != 10: {left.shape}"
        assert freq == 5.0
        assert qdur == 5.0
        print(f"  OK  left={left.shape}  right={right.shape}")
        return True
    except Exception as e:
        print(f"  FAIL: BridgeHandler test: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Smoke test Bridge-compatible HDF5 files")
    parser.add_argument("--hdf5", type=str, help="Path to a single HDF5 file")
    parser.add_argument("--dir", type=str, help="Path to directory of HDF5 files")
    parser.add_argument("--handler-test", action="store_true",
                        help="Also run BridgeHandler.build_left_right() test")
    args = parser.parse_args()

    paths = []
    if args.hdf5:
        paths.append(args.hdf5)
    if args.dir:
        d = Path(args.dir)
        paths.extend(sorted(str(p) for p in d.glob("*.hdf5")))

    if not paths:
        print("No files to check. Use --hdf5 or --dir.")
        sys.exit(1)

    all_ok = True
    for p in paths:
        ok = verify_episode(p)
        if ok and args.handler_test:
            ok = verify_with_bridge_handler(p)
        if not ok:
            all_ok = False

    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'} ({len(paths)} files)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
