"""Kaggle GPU: Paper B IDH 2.5D ResNet on UTSW (patient-level folds).

Datasets expected under /kaggle/input:
  - suvadipchakraborty/utsw-glioma-2d-mri-dataset  (PNG slices)
  - optional: itxsamad/btlab-paper-b-splits (folds CSV) else clone from GitHub
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/itxsamad1/brain-tumor-radiogenomics.git"
REPO_DIR = Path("/kaggle/working/brain-tumor-radiogenomics")


def find_data_root() -> Path:
    for cand in Path("/kaggle/input").rglob("BT0001"):
        if cand.is_dir():
            # parent of patient dirs
            return cand.parent
    raise FileNotFoundError("UTSW 2D patient folders not found under /kaggle/input")


def find_folds() -> Path:
    for p in Path("/kaggle/input").rglob("utsw_paper_b_folds.csv"):
        return p
    local = REPO_DIR / "data" / "splits" / "utsw_paper_b_folds.csv"
    if local.exists():
        return local
    raise FileNotFoundError("utsw_paper_b_folds.csv not found")


def main() -> None:
    import torch

    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    if not REPO_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)])
    else:
        subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
    os.chdir(REPO_DIR)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".[torch]"])
    sys.path.insert(0, str(REPO_DIR / "src"))

    data_root = find_data_root()
    folds = find_folds()
    print("data_root", data_root)
    print("folds", folds)

    fold = int(os.environ.get("FOLD", "1"))
    epochs = int(os.environ.get("EPOCHS", "20"))
    out = Path("/kaggle/working") / f"fold{fold}_idh_2p5d"
    cmd = [
        sys.executable,
        "-m",
        "btlab.train.paper_b_idh_2p5d",
        "--data-root",
        str(data_root),
        "--folds",
        str(folds),
        "--out-dir",
        str(out),
        "--fold",
        str(fold),
        "--epochs",
        str(epochs),
        "--batch-size",
        os.environ.get("BATCH_SIZE", "4"),
        "--n-slices",
        os.environ.get("N_SLICES", "5"),
    ]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Done", out)


if __name__ == "__main__":
    main()
