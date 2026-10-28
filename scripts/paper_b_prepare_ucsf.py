#!/usr/bin/env python3
"""Prepare UCSF-PDGM Paper B labels + IDH-stratified patient folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def normalize_idh(s: pd.Series) -> pd.Series:
    """UCSF uses wildtype vs specific mutations (IDH1 p.R132H, mutated (NOS), …)."""
    out = []
    for v in s.astype(str).str.strip():
        low = v.lower()
        if low in ("", "nan", "none", "na"):
            out.append(np.nan)
        elif "wild" in low:
            out.append(0)
        else:
            # any explicit mutation string counts as mutant
            out.append(1)
    return pd.Series(out, index=s.index, dtype=float)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", type=Path, default=Path("data/raw/ucsf-pdgm/UCSF-PDGM-metadata.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--exp-dir", type=Path, default=Path("experiments/paper_b"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw = pd.read_csv(args.meta)
    df = pd.DataFrame(
        {
            "patient_id": raw["ID"].astype(str),
            "sex": raw["Sex"],
            "sex_male": raw["Sex"].astype(str).str.upper().str.startswith("M").astype(int),
            "age": pd.to_numeric(raw["Age at MRI"], errors="coerce"),
            "grade": pd.to_numeric(raw["WHO CNS Grade"], errors="coerce"),
            "diagnosis": raw.get("Final pathologic diagnosis (WHO 2021)"),
            "idh": normalize_idh(raw["IDH"]),
            "codel_raw": raw.get("1p/19q"),
            "mgmt_raw": raw.get("MGMT status"),
            "brats21_id": raw.get("BraTS21 ID"),
            "brats21_seg_cohort": raw.get("BraTS21 Segmentation Cohort"),
            "os": pd.to_numeric(raw.get("OS"), errors="coerce"),
            "dead": pd.to_numeric(raw.get("1-dead 0-alive"), errors="coerce"),
        }
    )
    # Overlap flag: any BraTS21 ID present → exclude from BraTS-pretrained external claims later
    df["in_brats21"] = df["brats21_id"].notna() & (df["brats21_id"].astype(str).str.len() > 0)

    labeled = df.dropna(subset=["idh"]).copy()
    y = labeled["idh"].astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold = np.zeros(len(labeled), dtype=int)
    for i, (_, val_idx) in enumerate(skf.split(labeled["patient_id"], y), start=1):
        fold[val_idx] = i
    labeled["fold"] = fold

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.out_dir / "ucsf_paper_b_labels.csv"
    folds_path = args.out_dir / "ucsf_paper_b_folds.csv"
    df.to_csv(labels_path, index=False)
    labeled.to_csv(folds_path, index=False)

    summary = {
        "n_patients": int(len(df)),
        "n_idh_labeled": int(labeled.shape[0]),
        "idh_mutant": int((labeled["idh"] == 1).sum()),
        "idh_wildtype": int((labeled["idh"] == 0).sum()),
        "n_in_brats21": int(df["in_brats21"].sum()),
        "source": "UCSF-PDGM metadata (cite TCIA DOI 10.7937/tcia.bdgf-8v37)",
        "seed": args.seed,
    }
    (args.exp_dir / "ucsf_label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote", labels_path, folds_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
