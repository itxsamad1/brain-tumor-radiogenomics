"""Paper B: train IDH classifier on tumor-centered multi-modal crops (.npy).

Expects crops_index.csv from scripts/paper_b_make_crops.py.
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
from tqdm import tqdm


class CropDataset(Dataset):
    def __init__(self, table: pd.DataFrame, root: Path, train: bool):
        self.table = table.reset_index(drop=True)
        self.root = Path(root)
        self.train = train

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        x = np.load(self.root / row["path"]).astype(np.float32)  # C,D,H,W
        if self.train and np.random.rand() < 0.5:
            x = np.flip(x, axis=-1).copy()
        # pad/truncate channels to 4
        c = x.shape[0]
        if c < 4:
            x = np.concatenate([x, np.zeros((4 - c, *x.shape[1:]), dtype=np.float32)], axis=0)
        elif c > 4:
            x = x[:4]
        y = int(row["idh"])
        clin = np.array(
            [
                (float(row.get("age", 50.0)) - 50.0) / 15.0,
                float(row.get("sex_male", 0.0)),
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(x), torch.tensor(y), torch.from_numpy(clin)


class Small3DCNN(nn.Module):
    def __init__(self, in_ch: int = 4, fuse_clinical: bool = True):
        super().__init__()
        self.fuse = fuse_clinical
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(64 + (2 if fuse_clinical else 0), 1)

    def forward(self, x, clin=None):
        f = self.enc(x).flatten(1)
        if self.fuse:
            f = torch.cat([f, clin], dim=1)
        return self.head(f).squeeze(1)


@torch.no_grad()
def eval_model(model, loader, device, fuse: bool):
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    model.eval()
    ys, ps = [], []
    for x, y, clin in loader:
        x, clin = x.to(device), clin.to(device)
        logit = model(x, clin) if fuse else model(x)
        prob = torch.sigmoid(logit).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--crops-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fuse-clinical", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--patience", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    idx = pd.read_csv(args.crops_root / "crops_index.csv").dropna(subset=["idh"])
    idx["idh"] = idx["idh"].astype(int)
    train_df = idx[idx["fold"] != args.fold].reset_index(drop=True)
    val_df = idx[idx["fold"] == args.fold].reset_index(drop=True)

    train_loader = DataLoader(
        CropDataset(train_df, args.crops_root, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        CropDataset(val_df, args.crops_root, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = Small3DCNN(fuse_clinical=args.fuse_clinical).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos = max(1, int((train_df["idh"] == 1).sum()))
    neg = max(1, int((train_df["idh"] == 0).sum()))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history, best, bad = [], -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for x, y, clin in tqdm(train_loader, desc=f"crop fold{args.fold} ep{epoch+1}"):
            x, y, clin = x.to(device), y.float().to(device), clin.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x, clin) if args.fuse_clinical else model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            n += len(y)
        metrics = eval_model(model, val_loader, device, args.fuse_clinical)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), **metrics}
        history.append(row)
        print(row)
        if metrics["auroc"] is not None and metrics["auroc"] > best:
            best = metrics["auroc"]
            bad = 0
            torch.save({"model": model.state_dict(), "best_auroc": best, "history": history}, args.out_dir / "checkpoint_best.pt")
        else:
            bad += 1
        (args.out_dir / "status.json").write_text(json.dumps({"epoch": epoch, "best_auroc": best, **metrics}, indent=2), encoding="utf-8")
        if bad >= args.patience:
            break
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("best_auroc", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
