#!/usr/bin/env python3
"""Paper B must-beat radiomics baselines with honest clinical comparison.

Runs patient-level 5-fold CV:
  - age+sex LR
  - radiomics-only LR
  - radiomics+clinical LR
  - radiomics+clinical XGBoost (if installed)

Reports mean AUROC/AUPRC/balanced accuracy and whether imaging beats age.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from btlab.data.nifti_cohorts import resolve_ucsf_case, resolve_utsw_case
from btlab.data.overlap import assert_no_patient_leak
from btlab.radiomics.features import extract_case_features


def _metrics(y, prob) -> dict:
    pred = (prob >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None,
        "auprc": float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float((pred == y).mean()),
    }


def build_feature_table(folds: pd.DataFrame, data_root: Path, cohort: str) -> pd.DataFrame:
    resolver = resolve_utsw_case if cohort == "utsw" else resolve_ucsf_case
    rows = []
    for _, r in folds.iterrows():
        pid = str(r["patient_id"])
        paths = resolver(data_root, pid)
        if paths is None:
            continue
        images = {k: paths[k] for k in ("t1c", "flair", "t2", "t1") if k in paths}
        try:
            feats = extract_case_features(images, paths["mask"])
        except Exception as e:  # noqa: BLE001 — keep batch robust
            print(f"skip {pid}: {e}")
            continue
        feats["patient_id"] = pid
        feats["idh"] = int(r["idh"])
        feats["fold"] = int(r["fold"])
        feats["age"] = float(r.get("age", np.nan))
        feats["sex_male"] = float(r.get("sex_male", 0))
        rows.append(feats)
        if len(rows) % 25 == 0:
            print(f"extracted {len(rows)} cases…")
    return pd.DataFrame(rows)


def eval_model(name: str, pipe, X_cols: list[str], df: pd.DataFrame) -> dict:
    fold_rows = []
    for fold in sorted(df["fold"].unique()):
        train = df[df["fold"] != fold]
        val = df[df["fold"] == fold]
        assert_no_patient_leak(train["patient_id"], val["patient_id"])
        pipe.fit(train[X_cols], train["idh"].astype(int))
        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(val[X_cols])[:, 1]
        else:
            prob = pipe.decision_function(val[X_cols])
            prob = 1 / (1 + np.exp(-prob))
        m = _metrics(val["idh"].astype(int).to_numpy(), np.asarray(prob))
        m["fold"] = int(fold)
        m["n_val"] = int(len(val))
        fold_rows.append(m)
        print(name, "fold", fold, m)
    res = pd.DataFrame(fold_rows)
    return {
        "model": name,
        "features": X_cols,
        "n_folds": int(len(res)),
        "auroc_mean": float(res["auroc"].mean()),
        "auroc_std": float(res["auroc"].std(ddof=0)),
        "auprc_mean": float(res["auprc"].mean()),
        "balanced_accuracy_mean": float(res["balanced_accuracy"].mean()),
        "folds": fold_rows,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=["utsw", "ucsf"], required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--features-cache", type=Path, default=None, help="Reuse/save extracted features CSV")
    p.add_argument("--max-patients", type=int, default=0, help="Debug subsample (0=all)")
    args = p.parse_args()

    folds = pd.read_csv(args.folds).dropna(subset=["idh"]).copy()
    folds["idh"] = folds["idh"].astype(int)
    if args.max_patients > 0:
        # keep fold balance roughly
        folds = folds.groupby("fold", group_keys=False).head(max(1, args.max_patients // 5))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.features_cache or (args.out_dir / f"{args.cohort}_radiomics_features.csv")
    if cache.exists():
        feat_df = pd.read_csv(cache)
        print("loaded features", cache, len(feat_df))
    else:
        feat_df = build_feature_table(folds, args.data_root, args.cohort)
        feat_df.to_csv(cache, index=False)
        print("wrote", cache, len(feat_df))

    if len(feat_df) < 50:
        raise SystemExit(f"Too few extracted cases ({len(feat_df)}); check --data-root layout")

    # drop non-feature cols
    meta = {"patient_id", "idh", "fold", "age", "sex_male"}
    rad_cols = [c for c in feat_df.columns if c not in meta]
    # replace inf
    feat_df[rad_cols] = feat_df[rad_cols].replace([np.inf, -np.inf], np.nan)

    summaries = []
    clin_cols = ["age", "sex_male"]
    summaries.append(
        eval_model(
            "age_sex_lr",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, class_weight="balanced")),
                ]
            ),
            clin_cols,
            feat_df,
        )
    )
    summaries.append(
        eval_model(
            "radiomics_lr",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5)),
                ]
            ),
            rad_cols,
            feat_df,
        )
    )
    summaries.append(
        eval_model(
            "radiomics_clinical_lr",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", C=0.5)),
                ]
            ),
            rad_cols + clin_cols,
            feat_df,
        )
    )

    try:
        from xgboost import XGBClassifier

        # scale_pos_weight from training prevalence approximated globally
        pos = max(1, int((feat_df["idh"] == 1).sum()))
        neg = max(1, int((feat_df["idh"] == 0).sum()))
        summaries.append(
            eval_model(
                "radiomics_clinical_xgb",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        (
                            "clf",
                            XGBClassifier(
                                n_estimators=300,
                                max_depth=3,
                                learning_rate=0.05,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                reg_lambda=1.0,
                                objective="binary:logistic",
                                eval_metric="auc",
                                n_jobs=4,
                                scale_pos_weight=neg / pos,
                            ),
                        ),
                    ]
                ),
                rad_cols + clin_cols,
                feat_df,
            )
        )
    except Exception as e:  # noqa: BLE001
        print("XGBoost skipped:", e)

    age = next(s for s in summaries if s["model"] == "age_sex_lr")
    for s in summaries:
        s["delta_auroc_vs_age"] = (
            None if s["auroc_mean"] is None else float(s["auroc_mean"] - age["auroc_mean"])
        )
        s["beats_age"] = bool(s["delta_auroc_vs_age"] is not None and s["delta_auroc_vs_age"] > 0)

    out = {
        "cohort": args.cohort,
        "n_patients_features": int(len(feat_df)),
        "n_radiomics_features": int(len(rad_cols)),
        "models": summaries,
        "protocol_note": (
            "Patient-level 5-fold CV on extracted cases only. "
            "Confirmatory external evaluation must follow docs/analysis_plan.md lock."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({m["model"]: {k: m[k] for k in ("auroc_mean", "delta_auroc_vs_age", "beats_age")} for m in summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
