"""Classification metrics for Paper A (patient-level reporting)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(m):
            continue
        acc = (pred[m] == y_true[m]).mean()
        ece += m.mean() * abs(acc - conf[m].mean())
    return float(ece)


def classification_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probs = softmax(np.asarray(logits))
    y_pred = probs.argmax(axis=1)
    n_classes = probs.shape[1]
    names = class_names or [str(i) for i in range(n_classes)]

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ece": expected_calibration_error(probs, y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist(),
        "per_class": {
            names[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
        "report": classification_report(
            y_true, y_pred, target_names=names, zero_division=0, digits=4
        ),
    }
    # multiclass OVR AUROC / AUPRC
    try:
        out["auroc_macro_ovr"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        )
    except ValueError:
        out["auroc_macro_ovr"] = float("nan")
    try:
        y_bin = np.eye(n_classes)[y_true]
        out["auprc_macro"] = float(
            average_precision_score(y_bin, probs, average="macro")
        )
    except ValueError:
        out["auprc_macro"] = float("nan")
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    logits: np.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = classification_metrics(y_true[idx], logits[idx])
        vals.append(m[metric])
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "low": float(np.nanpercentile(arr, 2.5)),
        "high": float(np.nanpercentile(arr, 97.5)),
    }
