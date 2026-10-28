#!/usr/bin/env python3
"""Export tumor-centered multi-modal float16 crops for Paper B Kaggle packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btlab.data.nifti_cohorts import resolve_ucsf_case, resolve_utsw_case
from btlab.radiomics.features import load_nifti, tumor_centered_crop


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=["utsw", "ucsf"], required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--max-patients", type=int, default=0)
    args = p.parse_args()

    resolver = resolve_utsw_case if args.cohort == "utsw" else resolve_ucsf_case
    df = pd.read_csv(args.folds).dropna(subset=["idh"])
    if args.max_patients > 0:
        df = df.head(args.max_patients)

    crops_dir = args.out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    size = (args.size, args.size, args.size)
    for _, r in df.iterrows():
        pid = str(r["patient_id"])
        paths = resolver(args.data_root, pid)
        if paths is None:
            continue
        mask, _ = load_nifti(paths["mask"])
        imgs = {}
        for mod in ("t1c", "flair", "t2", "t1"):
            if mod not in paths:
                continue
            vol, _ = load_nifti(paths[mod])
            if vol.shape != mask.shape:
                continue
            brain = vol[np.isfinite(vol) & (np.abs(vol) > 0)]
            if brain.size > 10:
                vol = (vol - brain.mean()) / (brain.std() + 1e-8)
            imgs[mod] = vol
        if "t1c" not in imgs:
            continue
        crops = tumor_centered_crop(imgs, mask, size=size)
        # stack modalities in fixed order
        order = [m for m in ("t1c", "flair", "t2", "t1") if m in crops]
        stack = np.stack([crops[m] for m in order], axis=0).astype(np.float16)
        out_fp = crops_dir / f"{pid}.npy"
        np.save(out_fp, stack)
        rows.append(
            {
                "patient_id": pid,
                "path": str(out_fp.relative_to(args.out_dir)),
                "idh": int(r["idh"]),
                "fold": int(r["fold"]),
                "age": r.get("age"),
                "sex_male": r.get("sex_male"),
                "modalities": ",".join(order),
                "shape": "x".join(map(str, stack.shape)),
            }
        )
        if len(rows) % 20 == 0:
            print("crops", len(rows))

    meta = pd.DataFrame(rows)
    meta.to_csv(args.out_dir / "crops_index.csv", index=False)
    (args.out_dir / "crop_meta.json").write_text(
        json.dumps(
            {
                "cohort": args.cohort,
                "n": int(len(meta)),
                "size": list(size),
                "dtype": "float16",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", len(meta), "crops ->", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
