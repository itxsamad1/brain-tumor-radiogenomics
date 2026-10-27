#!/usr/bin/env python3
"""Export Figshare mats to PNG + package folder for Kaggle Dataset upload."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from btlab.data.figshare import _load_mat


def mat_to_uint8(path: Path) -> np.ndarray:
    data = _load_mat(path)
    obj = data.get("cjdata", data)
    arr = np.array(getattr(obj, "image"), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] < 64:
        arr = arr.T
    arr = arr - arr.min()
    mx = arr.max()
    if mx > 0:
        arr = arr / mx
    return (arr * 255.0).astype(np.uint8)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mats-root", type=Path, default=Path("data/raw/figshare/extracted/mats"))
    p.add_argument("--splits", type=Path, default=Path("data/splits/figshare_cvind_patient.csv"))
    p.add_argument("--out", type=Path, default=Path("data/kaggle_upload/btlab-figshare-paper-a"))
    args = p.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    png_root = args.out / "png"
    png_root.mkdir(parents=True)

    splits = pd.read_csv(args.splits)
    rows = []
    for _, row in tqdm(splits.iterrows(), total=len(splits), desc="export png"):
        mat = args.mats_root / row["subset"] / row["file"]
        if not mat.exists():
            hits = list(args.mats_root.rglob(row["file"]))
            if not hits:
                raise FileNotFoundError(mat)
            mat = hits[0]
        img = mat_to_uint8(mat)
        rel = f"{row['subset']}/{Path(row['file']).stem}.png"
        out_png = png_root / rel
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_png)
        rows.append(
            {
                "patient_id": row["patient_id"],
                "subset": row["subset"],
                "file": Path(row["file"]).stem + ".png",
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "fold": int(row["fold"]),
            }
        )

    pd.DataFrame(rows).to_csv(args.out / "splits.csv", index=False)
    shutil.copy2(args.splits.with_suffix(args.splits.suffix + ".manifest.json"), args.out / "splits.manifest.json")
    (args.out / "README.md").write_text(
        "# btlab Figshare Paper A\n\n"
        "Source: https://doi.org/10.6084/m9.figshare.1512427 (CC BY 4.0). Cite Cheng et al.\n"
        "Patient-level folds in splits.csv. Images under png/.\n",
        encoding="utf-8",
    )
    print(f"Packed {len(rows)} PNGs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
