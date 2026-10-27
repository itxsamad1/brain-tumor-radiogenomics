#!/usr/bin/env python3
"""Paper A audits: split protocol comparison scaffolding + near-duplicate hash audit.

Tumor-masked and full slice-level training are launched separately on Kaggle;
this script produces the duplicate audit and a summary table from available results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def dhash(path: Path, size: int = 8) -> str:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = "".join("1" if b else "0" for b in diff.flatten())
    return hashlib.md5(bits.encode()).hexdigest()


def near_duplicate_audit(splits: pd.DataFrame, data_root: Path, out: Path) -> dict:
    """Flag identical dHash across different patients (possible near-duplicates)."""
    rows = []
    for _, r in tqdm(splits.iterrows(), total=len(splits), desc="dhash"):
        png = data_root / "png" / r["subset"] / r["file"]
        if not png.exists():
            png = data_root / "png" / r["file"]
        if not png.exists():
            continue
        rows.append({"patient_id": r["patient_id"], "fold": int(r["fold"]), "file": r["file"], "dhash": dhash(png)})
    df = pd.DataFrame(rows)
    # same hash, different patients
    bad = []
    for h, g in df.groupby("dhash"):
        pids = g["patient_id"].unique()
        if len(pids) > 1:
            bad.append({"dhash": h, "n_slices": len(g), "patients": ",".join(map(str, pids)), "folds": ",".join(map(str, sorted(g["fold"].unique())))})
    bad_df = pd.DataFrame(bad)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "slice_dhash.csv", index=False)
    bad_df.to_csv(out / "cross_patient_identical_dhash.csv", index=False)
    summary = {
        "n_slices_hashed": int(len(df)),
        "n_unique_hashes": int(df["dhash"].nunique()),
        "n_cross_patient_identical_hash_groups": int(len(bad_df)),
    }
    (out / "duplicate_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fold_balance_table(splits: pd.DataFrame, out: Path) -> None:
    g = (
        splits.groupby(["fold", "class_name"])
        .agg(slices=("file", "count"), patients=("patient_id", "nunique"))
        .reset_index()
    )
    g.to_csv(out / "fold_class_balance.csv", index=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=Path, default=Path("data/splits/figshare_cvind_patient.csv"))
    p.add_argument("--data-root", type=Path, default=Path("data/kaggle_upload/btlab-figshare-paper-a"))
    p.add_argument("--out-dir", type=Path, default=Path("experiments/paper_a/audits"))
    args = p.parse_args()

    # Prefer PNG splits if present
    png_splits = args.data_root / "splits.csv"
    splits = pd.read_csv(png_splits if png_splits.exists() else args.splits)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_balance_table(splits, args.out_dir)
    summary = near_duplicate_audit(splits, args.data_root, args.out_dir)
    print(json.dumps(summary, indent=2))

    md = [
        "# Paper A — Audits",
        "",
        "## Near-duplicate (dHash) audit",
        f"- Slices hashed: {summary['n_slices_hashed']}",
        f"- Unique hashes: {summary['n_unique_hashes']}",
        f"- Cross-patient identical hash groups: **{summary['n_cross_patient_identical_hash_groups']}**",
        "",
        "See `cross_patient_identical_dhash.csv` for details.",
        "",
        "## Fold balance",
        "See `fold_class_balance.csv`.",
        "",
        "## Still required on GPU",
        "- Slice-level (leaky) split training vs patient-level (main comparison figure).",
        "- Tumor-masked control experiment.",
        "- Grad-CAM examples.",
        "",
    ]
    (args.out_dir / "AUDIT_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
