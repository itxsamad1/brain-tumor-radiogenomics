"""Kaggle GPU: Paper B IDH 2.5D ResNet on UTSW (patient-level folds).

Env:
  FOLDS=1,2,3,4,5
  EPOCHS=20
  FUSE_CLINICAL=1   # late-fuse age+sex (default on for folds that must beat age LR)
  BATCH_SIZE=4
  N_SLICES=5
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
    folds_csv = find_folds()
    print("data_root", data_root)
    print("folds", folds_csv)

    fold_list = [int(x) for x in os.environ.get("FOLDS", "2,3,4,5").split(",") if x.strip()]
    epochs = int(os.environ.get("EPOCHS", "20"))
    fuse = os.environ.get("FUSE_CLINICAL", "1") not in ("0", "false", "False")

    for fold in fold_list:
        out = Path("/kaggle/working") / f"fold{fold}_idh_2p5d"
        # skip finished folds (resume-friendly)
        status = out / "status.json"
        if status.exists():
            import json

            st = json.loads(status.read_text())
            if st.get("best_auroc") is not None and st.get("epoch", 0) >= epochs - 1:
                print(f"skip fold {fold} already done best={st.get('best_auroc')}")
                continue
        cmd = [
            sys.executable,
            "-m",
            "btlab.train.paper_b_idh_2p5d",
            "--data-root",
            str(data_root),
            "--folds",
            str(folds_csv),
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
