"""Kaggle: Paper A full phase — train remaining folds + evaluate figures.

Env vars:
  FOLDS=1,2,3,4,5
  MODEL=resnet18
  EPOCHS=30
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/itxsamad1/brain-tumor-radiogenomics.git"
REPO_DIR = Path("/kaggle/working/brain-tumor-radiogenomics")
WORK = Path("/kaggle/working")


def find_data_root() -> Path:
    for p in Path("/kaggle/input").rglob("splits.csv"):
        if (p.parent / "png").exists() or any(p.parent.glob("png")):
            return p.parent
    for p in Path("/kaggle/input").rglob("splits.csv"):
        return p.parent
    raise FileNotFoundError("splits.csv not found")


def main() -> None:
    import torch

    print("cuda", torch.cuda.is_available())
    if not REPO_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)])
    else:
        subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
    os.chdir(REPO_DIR)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "."])
    sys.path.insert(0, str(REPO_DIR / "src"))

    data_root = find_data_root()
    splits = data_root / "splits.csv"
    folds = [int(x) for x in os.environ.get("FOLDS", "2,3,4,5").split(",") if x.strip()]
    model = os.environ.get("MODEL", "resnet18")
    epochs = int(os.environ.get("EPOCHS", "30"))

    # seed prior ckpts
    for src in Path("/kaggle/input").rglob("checkpoint_best.pt"):
        fold_guess = src.parent.name
        dest = WORK / "checkpoints" / fold_guess
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / "checkpoint_best.pt").exists():
            shutil.copy2(src, dest / src.name)

    for fold in folds:
        out = WORK / "checkpoints" / f"fold{fold}_{model}"
        out.mkdir(parents=True, exist_ok=True)
        for pattern in ("checkpoint_last.pt", "checkpoint_best.pt"):
            for src in Path("/kaggle/input").rglob(pattern):
                if f"fold{fold}" in str(src) or f"fold_{fold}" in str(src):
                    shutil.copy2(src, out / pattern)
        cmd = [
            sys.executable, "-m", "btlab.train.paper_a",
            "--data-root", str(data_root),
            "--splits", str(splits),
            "--out-dir", str(out),
            "--fold", str(fold),
            "--epochs", str(epochs),
            "--model", model,
            "--use-png",
            "--workers", "2",
            "--resume-from", str(out),
        ]
        print("RUN", " ".join(cmd))
        subprocess.check_call(cmd)

        # evaluate into working
        eval_out = WORK / "eval" / f"fold{fold}_{model}"
        best = out / "checkpoint_best.pt"
        if best.exists():
            subprocess.check_call([
                sys.executable, "scripts/paper_a_evaluate.py",
                "--checkpoint", str(best),
                "--data-root", str(data_root),
                "--splits", str(splits),
                "--out-dir", str(eval_out),
                "--fold", str(fold),
                "--use-png",
                "--bootstrap", "500",
            ])

    # copy artifacts to /kaggle/working root
    for p in (WORK / "checkpoints").rglob("*"):
        if p.suffix in {".pt", ".json"}:
            rel = p.relative_to(WORK / "checkpoints")
            dest = WORK / "artifacts" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    print("DONE Paper A multi-fold")


if __name__ == "__main__":
    main()
