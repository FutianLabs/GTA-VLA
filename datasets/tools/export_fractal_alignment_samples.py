#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from pathlib import Path

from datasets.dataset import InfiniteDataReader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metas_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--action_mode", default="ee6d")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--dataset_name", default="Fractal")
    parser.add_argument("--force_fractal_cot", action="store_true")
    parser.add_argument("--annotation_dir", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["XVLA_EXPORT_ALIGN_DIR"] = str(out_dir)
    os.environ["XVLA_EXPORT_ALIGN_LIMIT"] = str(args.max_samples)
    os.environ["XVLA_EXPORT_ALIGN_DATASET"] = args.dataset_name

    metas_path = args.metas_path
    temp_meta_path = None
    model_config = None
    if args.force_fractal_cot:
        meta_path = Path(args.metas_path).expanduser().resolve()
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("dataset_name") != "Fractal":
            raise ValueError("force_fractal_cot only supports dataset_name=Fractal")
        ann = args.annotation_dir.strip() or meta.get("annotation_dir", "")
        if not ann:
            raise ValueError("force_fractal_cot requires annotation_dir")
        meta["annotation_dir"] = ann
        fd, tmp = tempfile.mkstemp(prefix="fractal_cot_meta_", suffix=".json")
        os.close(fd)
        temp_meta_path = Path(tmp)
        with temp_meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        metas_path = str(temp_meta_path)
        model_config = {"use_cot_training": True}

    ds = InfiniteDataReader(
        metas_path=metas_path,
        num_actions=args.num_actions,
        training=False,
        action_mode=args.action_mode,
        image_color_jitter=False,
        image_processor=None,
        model_config=model_config,
    )

    count = 0
    for _ in ds:
        count += 1
        if count >= args.max_samples:
            break

    samples_path = out_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"export failed, missing {samples_path}")

    with samples_path.open("r", encoding="utf-8") as f:
        exported = sum(1 for _ in f)

    summary = {
        "requested_max_samples": args.max_samples,
        "iterated_samples": count,
        "exported_samples": exported,
        "samples_jsonl": str(samples_path),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if temp_meta_path is not None and temp_meta_path.exists():
        temp_meta_path.unlink()

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
