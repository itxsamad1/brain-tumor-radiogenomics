"""Paper B: patient-level 2.5D IDH classifier on multi-contrast MRI slices.

Designed for Kaggle GPU with UTSW 2D PNG export or on-the-fly NIfTI slicing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


MODALITIES = ("brain_t1ce", "brain_flair", "brain_t2", "brain_t1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UTSW2DPatientDataset(Dataset):
    """Load a few mid-tumor / mid-volume PNG slices per patient as a stacked tensor.

    Expected layout (Kaggle 2D pack):
      root/UTSW_Glioma_2D/dataset/<patient_id>/<modality>/slice_XXX.png
    """

    def __init__(
        self,
        table: pd.DataFrame,
        root: Path,
        train: bool,
        n_slices: int = 5,
        img_size: int = 224,
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.root = Path(root)
        self.train = train
        self.n_slices = n_slices
        self.img_size = img_size
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    def __len__(self) -> int:
        return len(self.table)

    def _patient_dir(self, pid: str) -> Path:
        candidates = [
            self.root / "UTSW_Glioma_2D" / "dataset" / pid,
            self.root / "dataset" / pid,
            self.root / pid,
        ]
        for c in candidates:
            if c.exists():
                return c
        # fuzzy search
        hits = list(self.root.rglob(pid))
        for h in hits:
            if h.is_dir():
                return h
        raise FileNotFoundError(f"No directory for patient {pid} under {self.root}")

    def _load_modality_stack(self, pdir: Path, modality: str) -> torch.Tensor:
        from PIL import Image

        mdir = pdir / modality
        if not mdir.exists():
            # try alternate names
            alts = {
                "brain_t1ce": ["brain_t1ce", "t1ce", "t1gd", "T1CE"],
                "brain_flair": ["brain_flair", "flair", "FLAIR", "brain_fl"],
                "brain_t2": ["brain_t2", "t2", "T2"],
                "brain_t1": ["brain_t1", "t1", "T1"],
            }
            for a in alts.get(modality, [modality]):
                if (pdir / a).exists():
                    mdir = pdir / a
                    break
        files = sorted(mdir.glob("*.png")) if mdir.exists() else []
        if not files:
            return torch.zeros(self.n_slices, 1, self.img_size, self.img_size)
        # take center slices
        mid = len(files) // 2
        half = self.n_slices // 2
        idxs = list(range(max(0, mid - half), min(len(files), mid - half + self.n_slices)))
        while len(idxs) < self.n_slices:
            idxs.append(idxs[-1] if idxs else 0)
        imgs = []
        for i in idxs[: self.n_slices]:
            arr = np.array(Image.open(files[i]).convert("L"), dtype=np.uint8)
            imgs.append(self.tf(arr))
        return torch.stack(imgs, dim=0)  # S,1,H,W

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        pdir = self._patient_dir(str(row["patient_id"]))
        mods = []
        for m in MODALITIES:
            mods.append(self._load_modality_stack(pdir, m))
        # S, 4, H, W  (stack modalities as channels after averaging? keep 4ch via mean over slices later)
        x = torch.stack(mods, dim=2)  # S,1,4,H,W? wait mods are S,1,H,W each
        # mods list of S,1,H,W -> stack channel: S,4,H,W
        x = torch.cat(mods, dim=1)  # S,4,H,W
        y = int(row["idh"])
        return x, y, str(row["patient_id"])


class SlicePoolResNet(nn.Module):
    def __init__(self, in_ch: int = 4, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        m = models.resnet18(weights=weights)
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.backbone = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,S,C,H,W
        b, s, c, h, w = x.shape
        logits = self.backbone(x.reshape(b * s, c, h, w)).reshape(b, s)
        return logits.mean(dim=1)


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y, _ in loader:
        x = x.to(device)
        logit = model(x)
        prob = torch.sigmoid(logit).cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    pred = (p >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "acc": float((pred == y).mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("out_paper_b"))
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-slices", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    df = pd.read_csv(args.folds).dropna(subset=["idh"])
    df["idh"] = df["idh"].astype(int)
    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    val_df = df[df["fold"] == args.fold].reset_index(drop=True)

    train_loader = DataLoader(
        UTSW2DPatientDataset(train_df, args.data_root, train=True, n_slices=args.n_slices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        UTSW2DPatientDataset(val_df, args.data_root, train=False, n_slices=args.n_slices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = SlicePoolResNet(in_ch=4, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # class imbalance
    n_pos = max(1, int((train_df["idh"] == 1).sum()))
    n_neg = max(1, int((train_df["idh"] == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for x, y, _ in tqdm(train_loader, desc=f"fold{args.fold} ep{epoch+1}"):
            x = x.to(device)
            y = y.float().to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            n += len(y)
        metrics = eval_loader(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), **metrics}
        history.append(row)
        print(row)
        if metrics["auroc"] is not None and metrics["auroc"] > best:
            best = metrics["auroc"]
            torch.save(
                {"model": model.state_dict(), "history": history, "best_auroc": best, "args": vars(args)},
                args.out_dir / "checkpoint_best.pt",
            )
        torch.save(
            {"model": model.state_dict(), "history": history, "epoch": epoch, "args": vars(args)},
            args.out_dir / "checkpoint_last.pt",
        )
        (args.out_dir / "status.json").write_text(
            json.dumps({"epoch": epoch, "best_auroc": best, **metrics}, indent=2),
            encoding="utf-8",
        )
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("best_auroc", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
