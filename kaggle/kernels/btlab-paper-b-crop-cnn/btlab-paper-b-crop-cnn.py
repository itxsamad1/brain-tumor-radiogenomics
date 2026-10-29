"""Kaggle GPU: Paper B tumor-crop 3D CNN on UCSF development folds.

Reads crops from kernel output of itxsamad/btlab-paper-b-crops.
Env:
  FOLDS=1,2,3,4,5
  EPOCHS=30
  FUSE_CLINICAL=1
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/itxsamad1/brain-tumor-radiogenomics.git"
REPO_DIR = Path("/kaggle/working/brain-tumor-radiogenomics")


def find_crops_root() -> Path:
    for p in Path("/kaggle/input").rglob("crops_index.csv"):
        return p.parent
    raise FileNotFoundError("crops_index.csv not found (need btlab-paper-b-crops output)")


def main() -> None:
    import torch

    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    if not REPO_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)])
    else:
        subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
    os.chdir(REPO_DIR)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".[torch]"])

    crops_root = find_crops_root()
    print("crops_root", crops_root)
    fold_list = [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",") if x.strip()]
    epochs = int(os.environ.get("EPOCHS", "30"))
    fuse = os.environ.get("FUSE_CLINICAL", "1") not in ("0", "false", "False")

    for fold in fold_list:
        out = Path("/kaggle/working") / f"fold{fold}_crop_cnn"
        cmd = [
            sys.executable,
            "-m",
            "btlab.train.paper_b_crop_cnn",
            "--crops-root",
            str(crops_root),
            "--out-dir",
            str(out),
            "--fold",
            str(fold),
            "--epochs",
            str(epochs),
            "--batch-size",
            os.environ.get("BATCH_SIZE", "4"),
            "--patience",
            os.environ.get("PATIENCE", "8"),
        ]
        if fuse:
            cmd.append("--fuse-clinical")
        print("+", " ".join(cmd))
        subprocess.check_call(cmd)
        print("Done", out)


if __name__ == "__main__":
    main()
