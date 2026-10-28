#!/usr/bin/env python3
"""Aggregate Paper A fold metrics + leaky baseline into summary tables/figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from btlab.eval.plots import plot_metric_bars


def load_metrics(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("experiments/paper_a"))
    args = p.parse_args()
    root = args.root

    rows = []
    for fold in range(1, 6):
        for cand in [
            root / f"fold{fold}_resnet18" / "metrics.json",
            root / "kaggle_full" / "eval" / f"fold{fold}_resnet18" / "metrics.json",
        ]:
            m = load_metrics(cand)
            if m:
                rows.append({
                    "fold": fold,
                    "model": "resnet18",
                    "protocol": "patient_level",
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "auroc_macro_ovr": m.get("auroc_macro_ovr"),
                    "ece": m.get("ece"),
                    "source": str(cand),
                })
                break

    leaky_metrics = load_metrics(root / "leaky_slice_split" / "metrics.json")
    leaky = load_metrics(root / "leaky_slice_split" / "status.json") or {}
    if leaky_metrics:
        rows.append({
            "fold": 0,
            "model": "resnet18",
            "protocol": "slice_level_leaky",
            "accuracy": leaky_metrics.get("accuracy"),
            "macro_f1": leaky_metrics["macro_f1"],
            "balanced_accuracy": leaky_metrics.get("balanced_accuracy"),
            "auroc_macro_ovr": leaky_metrics.get("auroc_macro_ovr"),
            "ece": leaky_metrics.get("ece"),
            "source": str(root / "leaky_slice_split" / "metrics.json"),
        })
    elif leaky and "best_f1" in leaky:
        rows.append({
            "fold": 0,
            "model": "resnet18",
            "protocol": "slice_level_leaky",
            "accuracy": None,
            "macro_f1": leaky["best_f1"],
            "balanced_accuracy": None,
            "auroc_macro_ovr": None,
            "ece": None,
            "source": str(root / "leaky_slice_split" / "status.json"),
        })

    df = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "summary_metrics.csv", index=False)

    patient = df[df["protocol"] == "patient_level"]
    summary = {}
    if len(patient):
        summary["patient_level_resnet18"] = {
            "n_folds": int(len(patient)),
            "macro_f1_mean": float(patient["macro_f1"].mean()),
            "macro_f1_std": float(patient["macro_f1"].std(ddof=0)) if len(patient) > 1 else 0.0,
            "accuracy_mean": float(patient["accuracy"].mean()),
            "accuracy_std": float(patient["accuracy"].std(ddof=0)) if len(patient) > 1 else 0.0,
        }
    if leaky_metrics:
        summary["slice_level_leaky_macro_f1"] = leaky_metrics["macro_f1"]
        summary["slice_level_leaky_accuracy"] = leaky_metrics.get("accuracy")
    elif leaky and "best_f1" in leaky:
        summary["slice_level_leaky_macro_f1"] = leaky["best_f1"]
    leak_stats = load_metrics(root / "leaky_slice_split" / "leakage_stats.json")
    if leak_stats:
        summary["leakage_stats"] = leak_stats
    elif leaky.get("leakage_stats"):
        summary["leakage_stats"] = leaky["leakage_stats"]

    masked = load_metrics(root / "tumor_masked" / "eval_masked_metrics.json")
    if masked:
        summary["tumor_masked_eval_macro_f1"] = masked.get("macro_f1")
        summary["tumor_masked_eval_accuracy"] = masked.get("accuracy")
    masked_train = load_metrics(root / "tumor_masked" / "train_masked_metrics.json")
    if masked_train:
        summary["tumor_masked_train_best_macro_f1"] = masked_train.get("best_macro_f1")

    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # comparison bar if we have both
    labels, means = [], []
    if summary.get("patient_level_resnet18"):
        labels.append("patient-level\n(mean folds)")
        means.append(summary["patient_level_resnet18"]["macro_f1_mean"])
    if "slice_level_leaky_macro_f1" in summary:
        labels.append("slice-level\n(leaky)")
        means.append(summary["slice_level_leaky_macro_f1"])
    if "tumor_masked_eval_macro_f1" in summary:
        labels.append("tumor-zeroed\n(eval)")
        means.append(summary["tumor_masked_eval_macro_f1"])
    if labels:
        plot_metric_bars(
            labels, means, None, None,
            root / "figures" / "patient_vs_slice_macro_f1.png",
            title="Paper A: patient-level vs audits (macro-F1)",
            ylabel="macro-F1",
        )

    print(json.dumps(summary, indent=2))
    print("Wrote", root / "summary_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
