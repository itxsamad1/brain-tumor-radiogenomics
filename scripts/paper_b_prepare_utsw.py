#!/usr/bin/env python3
"""Prepare UTSW Paper B labels + frozen patient folds from official TSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from btlab.data.utsw import build_paper_b_labels, load_utsw_metadata, patient_stratified_folds


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--meta",
        type=Path,
        default=Path("data/raw/utsw-glioma/UTSW_Glioma_Metadata.tsv"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--exp-dir", type=Path, default=Path("experiments/paper_b"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    meta = load_utsw_metadata(args.meta)
    labels = build_paper_b_labels(meta)
    folds = patient_stratified_folds(labels, n_folds=args.n_folds, seed=args.seed, strata_col="idh")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.out_dir / "utsw_paper_b_labels.csv"
    folds_path = args.out_dir / "utsw_paper_b_folds.csv"
    labels.to_csv(labels_path, index=False)
    folds.to_csv(folds_path, index=False)

    summary = {
        "n_patients": int(len(labels)),
        "n_idh_labeled": int(labels["idh"].notna().sum()),
        "idh_mutant": int((labels["idh"] == 1).sum()),
        "idh_wildtype": int((labels["idh"] == 0).sum()),
        "codel_among_idhmut": int(labels.loc[labels["idh"] == 1, "codel"].notna().sum()),
        "folds": {
            str(i): {
                "n": int((folds["fold"] == i).sum()),
                "idh_mut_frac": float(folds.loc[folds["fold"] == i, "idh"].mean()),
            }
            for i in range(1, args.n_folds + 1)
        },
        "source_doi": "10.7937/DFAE-1B86",
        "seed": args.seed,
    }
    (args.exp_dir / "utsw_label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote", labels_path)
    print("Wrote", folds_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
