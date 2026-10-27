"""Figure generation for Paper A."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from btlab.eval.classification import softmax


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: list[str], out: Path, title: str = "Confusion matrix") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    _save(fig, out)


def plot_history(history: list[dict[str, Any]], out: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, [h["train_loss"] for h in history], marker="o", ms=3)
    ax[0].set_title("Train loss")
    ax[0].set_xlabel("Epoch")
    ax[1].plot(epochs, [h.get("macro_f1", np.nan) for h in history], marker="o", ms=3, label="macro-F1")
    ax[1].plot(epochs, [h.get("acc", np.nan) for h in history], marker="o", ms=3, label="acc")
    ax[1].set_title("Validation metrics")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    _save(fig, out)


def plot_roc_pr(y_true: np.ndarray, logits: np.ndarray, class_names: list[str], out_roc: Path, out_pr: Path) -> None:
    probs = softmax(logits)
    n_classes = probs.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("One-vs-rest ROC")
    ax.legend(fontsize=8)
    _save(fig, out_roc)

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(class_names):
        p, r, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ax.plot(r, p, label=f"{name} (AP={auc(r, p):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall")
    ax.legend(fontsize=8)
    _save(fig, out_pr)


def plot_reliability(y_true: np.ndarray, logits: np.ndarray, out: Path, n_bins: int = 10) -> None:
    probs = softmax(logits)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, counts = [], [], []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(m):
            continue
        bin_acc.append((pred[m] == y_true[m]).mean())
        bin_conf.append(conf[m].mean())
        counts.append(m.sum())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(bin_conf, bin_acc, "o-", label="model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    _save(fig, out)


def plot_metric_bars(
    labels: list[str],
    means: list[float],
    lows: list[float] | None,
    highs: list[float] | None,
    out: Path,
    title: str,
    ylabel: str = "Score",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, color="#4C72B0")
    if lows and highs:
        yerr = np.vstack([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    _save(fig, out)
