#!/usr/bin/env python3
"""Package float16 tumor crops + labels for Kaggle Dataset upload.

Expects a prepared cache directory (see docs/data_access.md). Does not upload
raw full-resolution NIfTI dumps.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory with float16 crops / masks (e.g. data/cache/crops)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Patient-level labels CSV",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        required=True,
        help="Frozen splits CSV/JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/kaggle_upload"),
        help="Staging folder to zip / upload",
    )
    parser.add_argument("--name", type=str, default="btlab-crops-v1")
    args = parser.parse_args()

    out = args.out / args.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    crops_dst = out / "crops"
    shutil.copytree(args.cache_dir, crops_dst)
    shutil.copy2(args.labels, out / "labels.csv")
    shutil.copy2(args.splits, out / Path(args.splits).name)

    meta = {
        "name": args.name,
        "note": "Float16 tumor-centered crops for Kaggle; cite original TCIA/Figshare DOIs.",
        "dtype": "float16",
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Staged Kaggle package at: {out}")
    print("Next: compress and upload via Kaggle UI or `kaggle datasets create/version`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
