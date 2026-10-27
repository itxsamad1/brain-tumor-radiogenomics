"""Kaggle kernel: Paper A Figshare training with auto-resume checkpoints.

Attach datasets:
  - itxsamad/btlab-figshare-paper-a  (images + splits)
  - itxsamad/btlab-paper-a-ckpts     (optional; previous checkpoints)

Every epoch saves checkpoint_last.pt under /kaggle/working/checkpoints.
If a prior checkpoint dataset is attached, training resumes automatically.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/itxsamad1/brain-tumor-radiogenomics.git"
REPO_DIR = Path("/kaggle/working/brain-tumor-radiogenomics")
CKPT_DIR = Path("/kaggle/working/checkpoints")
DATA_CANDIDATES = [
    Path("/kaggle/input/btlab-figshare-paper-a"),
    Path("/kaggle/input/btlab-figshare-paper-a/btlab-figshare-paper-a"),
]


def find_data_root() -> Path:
    for p in DATA_CANDIDATES:
        if (p / "splits.csv").exists():
            return p
        if (p / "png").exists() and (p / "splits.csv").exists():
            return p
    # fallback: search
    for p in Path("/kaggle/input").rglob("splits.csv"):
        return p.parent
    raise FileNotFoundError("Could not find btlab-figshare-paper-a splits.csv under /kaggle/input")


def main() -> None:
    print("=== btlab Paper A (Kaggle) ===")
    print("CUDA?", end=" ")
    try:
        import torch

        print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception as e:
        print("torch import failed", e)
        raise

    if not REPO_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)])
    else:
        subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])

    os.chdir(REPO_DIR)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "."])

    # Ensure package importable
    sys.path.insert(0, str(REPO_DIR / "src"))

    data_root = find_data_root()
    splits = data_root / "splits.csv"
    print("data_root=", data_root)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Seed working dir with previous checkpoints if present in inputs
    for pattern in ("checkpoint_last.pt", "checkpoint_best.pt", "status.json", "history.json"):
        for src in Path("/kaggle/input").rglob(pattern):
            dst = CKPT_DIR / src.name
            if not dst.exists():
                print(f"Seeding checkpoint from {src}")
                shutil.copy2(src, dst)

    fold = int(os.environ.get("FOLD", "1"))
    epochs = int(os.environ.get("EPOCHS", "30"))
    model = os.environ.get("MODEL", "resnet18")
    batch = int(os.environ.get("BATCH_SIZE", "32"))

    cmd = [
        sys.executable,
        "-m",
        "btlab.train.paper_a",
        "--data-root",
        str(data_root),
        "--splits",
        str(splits),
        "--out-dir",
        str(CKPT_DIR),
        "--fold",
        str(fold),
        "--epochs",
        str(epochs),
        "--model",
        model,
        "--batch-size",
        str(batch),
        "--use-png",
        "--workers",
        "2",
        "--resume-from",
        str(CKPT_DIR),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    # Copy key artifacts to /kaggle/working root for easy Output download
    for name in ("checkpoint_last.pt", "checkpoint_best.pt", "status.json", "history.json"):
        src = CKPT_DIR / name
        if src.exists():
            shutil.copy2(src, Path("/kaggle/working") / name)
            print("Wrote", Path("/kaggle/working") / name)

    print("=== DONE ===")
    print("Next resume: download Output -> version dataset itxsamad/btlab-paper-a-ckpts -> re-run kernel with that dataset attached.")


if __name__ == "__main__":
    main()
