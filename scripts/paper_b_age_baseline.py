#!/usr/bin/env python3
"""Paper B must-beat baseline: age (± sex) logistic regression for IDH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def metrics(y_true, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auroc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else None,
        "auprc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else None,
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=Path, default=Path("data/splits/utsw_paper_b_folds.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("experiments/paper_b/age_baseline"))
    p.add_argument("--features", type=str, default="age,sex_male", help="Comma-separated columns")
    args = p.parse_args()

    df = pd.read_csv(args.folds).dropna(subset=["idh"]).copy()
    df["idh"] = df["idh"].astype(int)
    feats = [c.strip() for c in args.features.split(",") if c.strip()]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feats)

    rows = []
    for fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != fold]
        val = df[df["fold"] == fold]
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        pipe.fit(train[feats], train["idh"])
        prob = pipe.predict_proba(val[feats])[:, 1]
        m = metrics(val["idh"].to_numpy(), prob)
        m["fold"] = int(fold)
        m["n_val"] = int(len(val))
        rows.append(m)
        print(f"fold {fold}:", m)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "fold_metrics.csv", index=False)
    summary = {
        "model": "logistic_regression",
        "features": feats,
        "n_folds": int(len(res)),
        "auroc_mean": float(res["auroc"].mean()),
        "auroc_std": float(res["auroc"].std(ddof=0)),
        "balanced_accuracy_mean": float(res["balanced_accuracy"].mean()),
        "auprc_mean": float(res["auprc"].mean()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
