#!/usr/bin/env python3
"""Tumor-masked control for Paper A.

Evaluates (and optionally fine-trains) ResNet-18 with the tumor region zeroed
using Figshare cjdata.tumorMask. A large drop vs unmasked metrics means the
model relies on tumor appearance; persistent high accuracy suggests shortcuts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from btlab.data.figshare import CLASS_NAMES, _load_mat
from btlab.eval.classification import bootstrap_ci, classification_metrics
from btlab.train.paper_a import IMG_SIZE, build_model, evaluate, set_seed


def _find_mat(mats_root: Path, stem: str, subset: str) -> Path | None:
    cand = mats_root / subset / f"{stem}.mat"
    if cand.exists():
        return cand
    hits = list(mats_root.rglob(f"{stem}.mat"))
    return hits[0] if hits else None


def _load_image_and_mask(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = _load_mat(mat_path)
    obj = data.get("cjdata", data)
    img = np.array(getattr(obj, "image"), dtype=np.float32)
    mask = np.array(getattr(obj, "tumorMask"), dtype=np.float32)
    if img.ndim == 2 and img.shape[0] < img.shape[1] and img.shape[0] < 64:
        img = img.T
    if mask.ndim == 2 and mask.shape[0] < mask.shape[1] and mask.shape[0] < 64:
        mask = mask.T
    if mask.shape != img.shape:
        # nearest resize via simple indexing
        from PIL import Image

        mask = np.array(
            Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST
            ),
            dtype=np.float32,
        )
    mask = (mask > 0).astype(np.float32)
    img = img - img.min()
    mx = img.max()
    if mx > 0:
        img = img / mx
    return (img * 255.0).astype(np.uint8), mask


class TumorMaskedDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        mats_root: Path,
        train: bool,
        mode: str = "zero_tumor",
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.mats_root = Path(mats_root)
        self.mode = mode
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        self.mask_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        stem = Path(str(row["file"])).stem
        mat = _find_mat(self.mats_root, stem, str(row.get("subset", "")))
        if mat is None:
            raise FileNotFoundError(f"No mat for {stem}")
        img, mask = _load_image_and_mask(mat)
        if self.mode == "zero_tumor":
            img = (img.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
        elif self.mode == "tumor_only":
            img = (img.astype(np.float32) * mask).astype(np.uint8)
        x = self.tf(img)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        y = int(row["label"]) - 1
        return x, y


def collect_logits(model, loader, device):
    model.eval()
    ys, logits_all = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="masked-eval"):
            x = x.to(device)
            logits_all.append(model(x).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(logits_all)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--mats-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/paper_a/tumor_masked"))
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--mode", choices=["zero_tumor", "tumor_only"], default="zero_tumor")
    p.add_argument("--train-epochs", type=int, default=0, help="If >0, also train from scratch on masked images")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    splits = pd.read_csv(args.splits)
    val_df = splits[splits["fold"] == args.fold].reset_index(drop=True)
    train_df = splits[splits["fold"] != args.fold].reset_index(drop=True)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # --- eval pretrained patient-level checkpoint under masking ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt.get("args", {}).get("model", "resnet18"), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    val_loader = DataLoader(
        TumorMaskedDataset(val_df, args.mats_root, train=False, mode=args.mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    y_true, logits = collect_logits(model, val_loader, device)
    names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    metrics = classification_metrics(y_true, logits, class_names=names)
    report_txt = metrics.pop("report")
    metrics["bootstrap_macro_f1"] = bootstrap_ci(y_true, logits, "macro_f1", n_boot=args.bootstrap)
    metrics["mode"] = args.mode
    metrics["protocol"] = "eval_pretrained_on_masked"
    metrics["fold"] = args.fold
    (out / "eval_masked_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "eval_masked_report.txt").write_text(report_txt, encoding="utf-8")
    print("masked eval macro_f1=", metrics["macro_f1"], "acc=", metrics["accuracy"])

    result = {"eval_pretrained_on_masked": metrics}

    if args.train_epochs > 0:
        train_loader = DataLoader(
            TumorMaskedDataset(train_df, args.mats_root, train=True, mode=args.mode),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        model = build_model("resnet18", pretrained=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        history = []
        best_f1 = -1.0
        for epoch in range(args.train_epochs):
            model.train()
            running, n = 0.0, 0
            for x, y in tqdm(train_loader, desc=f"masked-train {epoch+1}/{args.train_epochs}"):
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                running += loss.item() * y.size(0)
                n += y.size(0)
            m = evaluate(model, val_loader, device)
            row = {"epoch": epoch, "train_loss": running / max(n, 1), **m}
            history.append(row)
            print(row)
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                torch.save({"model": model.state_dict(), "history": history, "best_f1": best_f1}, out / "checkpoint_best.pt")
        train_metrics = {
            "best_macro_f1": best_f1,
            "mode": args.mode,
            "protocol": "train_from_scratch_on_masked",
            "fold": args.fold,
            "epochs": args.train_epochs,
        }
        (out / "train_masked_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        result["train_from_scratch_on_masked"] = train_metrics
        print("masked train best_f1=", best_f1)

    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Wrote", out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
