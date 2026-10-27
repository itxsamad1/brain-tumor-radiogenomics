"""Paper A: patient-level 2D tumor-type classification with resumable checkpoints.

Designed for Kaggle 12h GPU sessions: every epoch writes checkpoint_last.pt;
restarts auto-resume from last epoch.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from btlab.data.figshare import CLASS_NAMES, _load_mat


IMG_SIZE = 224
NUM_CLASSES = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(search_roots: list[Path]) -> Path | None:
    """Prefer checkpoint_last.pt under any search root (Kaggle input/working)."""
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(root.rglob("checkpoint_last.pt"))
        candidates.extend(root.rglob("last.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class FigshareSliceDataset(Dataset):
    """Load Figshare slices from .mat or pre-exported .png/.npy."""

    def __init__(
        self,
        table: pd.DataFrame,
        root: Path,
        train: bool,
        use_png: bool = False,
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.root = Path(root)
        self.use_png = use_png
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

    def __len__(self) -> int:
        return len(self.table)

    def _resolve_path(self, row: pd.Series) -> Path:
        if "path" in row and pd.notna(row.get("path")) and Path(str(row["path"])).exists():
            return Path(str(row["path"]))
        subset = row.get("subset", "")
        fname = row.get("file", "")
        if self.use_png:
            png = self.root / "png" / str(subset) / (Path(str(fname)).stem + ".png")
            if png.exists():
                return png
            # flat png layout
            flat = self.root / "png" / (Path(str(fname)).stem + ".png")
            if flat.exists():
                return flat
        cand = self.root / str(subset) / str(fname)
        if cand.exists():
            return cand
        # search by filename
        hits = list(self.root.rglob(str(fname)))
        if hits:
            return hits[0]
        raise FileNotFoundError(f"Cannot resolve slice: {row.to_dict()}")

    def _load_image(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".png":
            from PIL import Image

            arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
        elif path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32)
        else:
            data = _load_mat(path)
            obj = data.get("cjdata", data)
            arr = np.array(getattr(obj, "image"), dtype=np.float32)
            # HDF5 matlab often stores transposed
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1] and arr.shape[0] < 64:
                arr = arr.T
        # robust normalize to 0..255 uint8 for torchvision
        arr = arr - arr.min()
        mx = arr.max()
        if mx > 0:
            arr = arr / mx
        return (arr * 255.0).astype(np.uint8)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.table.iloc[idx]
        img = self._load_image(self._resolve_path(row))
        x = self.tf(img)
        # grayscale -> 3ch for ImageNet backbones
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        y = int(row["label"]) - 1  # 0,1,2
        return x, y


def build_model(name: str = "resnet18", pretrained: bool = True) -> nn.Module:
    name = name.lower()
    weights = "DEFAULT" if pretrained else None
    if name == "resnet18":
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == "resnet50":
        m = models.resnet50(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    preds_all: list[int] = []
    labels_all: list[int] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        preds_all.extend(pred.cpu().tolist())
        labels_all.extend(y.cpu().tolist())
    acc = correct / max(total, 1)
    # macro-F1
    f1s = []
    for c in range(NUM_CLASSES):
        tp = sum(p == c and t == c for p, t in zip(preds_all, labels_all))
        fp = sum(p == c and t != c for p, t in zip(preds_all, labels_all))
        fn = sum(p != c and t == c for p, t in zip(preds_all, labels_all))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return {"acc": acc, "macro_f1": float(np.mean(f1s))}


def save_ckpt(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")

    splits = pd.read_csv(args.splits)
    root = Path(args.data_root)
    fold = args.fold

    train_df = splits[splits["fold"] != fold]
    val_df = splits[splits["fold"] == fold]
    print(f"fold={fold} train_slices={len(train_df)} val_slices={len(val_df)} "
          f"train_patients={train_df.patient_id.nunique()} val_patients={val_df.patient_id.nunique()}")

    train_ds = FigshareSliceDataset(train_df, root, train=True, use_png=args.use_png)
    val_ds = FigshareSliceDataset(val_df, root, train=False, use_png=args.use_png)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    search = [out_dir, Path("/kaggle/input"), Path("/kaggle/working")]
    if args.resume_from:
        search.insert(0, Path(args.resume_from))
    ckpt_path = find_latest_checkpoint(search)
    start_epoch = 0
    best_f1 = -1.0
    history: list[dict[str, Any]] = []

    if ckpt_path is not None:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("best_f1", -1.0))
        history = list(ckpt.get("history", []))
        print(f"-> start_epoch={start_epoch} best_f1={best_f1:.4f}")
    else:
        print("No checkpoint found; training from scratch.")

    stop_file = Path(args.stop_file)
    for epoch in range(start_epoch, args.epochs):
        if stop_file.exists():
            print(f"STOP file found at {stop_file}; exiting after saving.")
            break
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * y.size(0)
            n += y.size(0)
            pbar.set_postfix(loss=running / max(n, 1))

        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), **metrics}
        history.append(row)
        print(f"epoch {epoch}: loss={row['train_loss']:.4f} acc={metrics['acc']:.4f} macro_f1={metrics['macro_f1']:.4f}")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "history": history,
            "args": vars(args),
            "fold": fold,
            "class_names": CLASS_NAMES,
        }
        save_ckpt(out_dir / "checkpoint_last.pt", state)
        # also write small JSON status for easy inspection without torch
        (out_dir / "status.json").write_text(
            json.dumps({"epoch": epoch, "best_f1": best_f1, **metrics}, indent=2), encoding="utf-8"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            state["best_f1"] = best_f1
            save_ckpt(out_dir / "checkpoint_best.pt", state)
            print(f"  new best macro_f1={best_f1:.4f}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Done. best_f1={best_f1:.4f} checkpoints in {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper A resumable Figshare training")
    p.add_argument("--data-root", type=str, required=True, help="Root with mats/ or png/")
    p.add_argument("--splits", type=str, required=True, help="CSV with patient_id,file,subset,label,fold")
    p.add_argument("--out-dir", type=str, default="experiments/paper_a/fold0")
    p.add_argument("--fold", type=int, default=1, help="Held-out fold (1..5)")
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-png", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume-from", type=str, default="", help="Extra dir to search for checkpoints")
    p.add_argument("--stop-file", type=str, default="STOP")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
