#!/usr/bin/env python3
"""Train a intentionally leaky slice-level split baseline for Paper A comparison.

Same ResNet-18 recipe as patient-level; only the split protocol changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from btlab.train.paper_a import FigshareSliceDataset, build_model, evaluate, save_ckpt, set_seed
import torch.nn as nn
from tqdm import tqdm


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True, help="Any figshare splits CSV (for paths/labels)")
    p.add_argument("--out-dir", type=Path, default=Path("experiments/paper_a/leaky_slice_split"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-png", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    df = pd.read_csv(args.splits)

    # LEAKY: random slice split ignoring patient_id
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=args.seed, stratify=df["label"].to_numpy()
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Measure leakage: patients appearing in both splits
    overlap = set(train_df.patient_id) & set(val_df.patient_id)
    leak_info = {
        "protocol": "slice_level_random",
        "n_train_slices": len(train_df),
        "n_val_slices": len(val_df),
        "n_patients_in_both_splits": len(overlap),
        "frac_val_slices_from_shared_patients": float(
            val_df.patient_id.isin(overlap).mean()
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "leakage_stats.json").write_text(json.dumps(leak_info, indent=2), encoding="utf-8")
    print(json.dumps(leak_info, indent=2))

    train_loader = DataLoader(
        FigshareSliceDataset(train_df, args.data_root, train=True, use_png=args.use_png),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        FigshareSliceDataset(val_df, args.data_root, train=False, use_png=args.use_png),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = build_model("resnet18", pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for x, y in tqdm(train_loader, desc=f"leaky epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * y.size(0)
            n += y.size(0)
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), **metrics}
        history.append(row)
        print(row)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "best_f1": best_f1,
            "history": history,
            "leakage_stats": leak_info,
            "args": vars(args),
        }
        save_ckpt(args.out_dir / "checkpoint_last.pt", state)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            state["best_f1"] = best_f1
            save_ckpt(args.out_dir / "checkpoint_best.pt", state)

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out_dir / "status.json").write_text(
        json.dumps({"best_f1": best_f1, "leakage_stats": leak_info}, indent=2), encoding="utf-8"
    )
    print(f"Done leaky baseline best_f1={best_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
