#!/usr/bin/env python3
"""Run full Paper A evaluation on a checkpoint: metrics, CIs, figures, REPORT snippet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from btlab.data.figshare import CLASS_NAMES
from btlab.eval.classification import bootstrap_ci, classification_metrics
from btlab.eval.plots import (
    plot_confusion,
    plot_history,
    plot_reliability,
    plot_roc_pr,
)
from btlab.train.paper_a import FigshareSliceDataset, build_model


def collect_logits(model, loader, device):
    model.eval()
    ys, logits_all, pids = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            if len(batch) == 2:
                x, y = batch
                pid = ["?"] * len(y)
            else:
                x, y, pid = batch
            x = x.to(device)
            logits = model(x).cpu().numpy()
            logits_all.append(logits)
            ys.append(y.numpy())
            pids.extend(list(pid))
    return np.concatenate(ys), np.concatenate(logits_all), pids


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/paper_a/fold1"))
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--use-png", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("args", {}).get("model", "resnet18")
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    splits = pd.read_csv(args.splits)
    val_df = splits[splits["fold"] == args.fold].reset_index(drop=True)
    ds = FigshareSliceDataset(val_df, args.data_root, train=False, use_png=args.use_png)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true, logits, _ = collect_logits(model, loader, device)
    names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    # labels in data are 1..3; model uses 0..2
    metrics = classification_metrics(y_true, logits, class_names=names)
    ci_f1 = bootstrap_ci(y_true, logits, "macro_f1", n_boot=args.bootstrap)
    ci_acc = bootstrap_ci(y_true, logits, "accuracy", n_boot=args.bootstrap)
    metrics["bootstrap_macro_f1"] = ci_f1
    metrics["bootstrap_accuracy"] = ci_acc

    out = args.out_dir
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # strip huge text for JSON
    report_txt = metrics.pop("report")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    np.savez_compressed(out / "predictions.npz", y_true=y_true, logits=logits)

    plot_confusion(np.array(metrics["confusion_matrix"]), names, fig_dir / "confusion_matrix.png",
                   title=f"Fold {args.fold} — {model_name}")
    plot_roc_pr(y_true, logits, names, fig_dir / "roc.png", fig_dir / "pr.png")
    plot_reliability(y_true, logits, fig_dir / "reliability.png")
    history = ckpt.get("history") or []
    if history:
        plot_history(history, fig_dir / "training_curves.png")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    md = [
        f"# Paper A — Fold {args.fold} evaluation",
        "",
        f"- Model: `{model_name}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Val slices: {len(val_df)} | patients: {val_df.patient_id.nunique()}",
        "",
        "## Metrics",
        f"- Accuracy: **{metrics['accuracy']:.4f}** (95% CI {ci_acc['low']:.4f}–{ci_acc['high']:.4f})",
        f"- Macro-F1: **{metrics['macro_f1']:.4f}** (95% CI {ci_f1['low']:.4f}–{ci_f1['high']:.4f})",
        f"- Balanced accuracy: **{metrics['balanced_accuracy']:.4f}**",
        f"- AUROC (macro OVR): **{metrics['auroc_macro_ovr']:.4f}**",
        f"- AUPRC (macro): **{metrics['auprc_macro']:.4f}**",
        f"- ECE: **{metrics['ece']:.4f}**",
        "",
        "## Per-class",
        "```",
        report_txt,
        "```",
        "",
        "## Figures",
        "- `figures/confusion_matrix.png`",
        "- `figures/roc.png`, `figures/pr.png`",
        "- `figures/reliability.png`",
        "- `figures/training_curves.png`",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote", out / "REPORT.md")
    print(report_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
