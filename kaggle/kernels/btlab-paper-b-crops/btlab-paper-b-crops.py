"""Kaggle: build tumor-centered float16 crops (UCSF or UTSW) for Paper B CNN training.

Env:
  COHORT=ucsf|utsw
  SIZE=96
  MAX_PATIENTS=0
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/itxsamad1/brain-tumor-radiogenomics.git"
REPO_DIR = Path("/kaggle/working/brain-tumor-radiogenomics")


def find_data_root(cohort: str) -> Path:
    root = Path("/kaggle/input")
    print("input tree (depth2):")
    for p in sorted(root.rglob("*"))[:80]:
        if len(p.relative_to(root).parts) <= 2:
            print(" ", p)
    if cohort == "ucsf":
        for hit in root.rglob("UCSF-PDGM-0004_nifti"):
            if hit.is_dir():
                return hit.parent
        for hit in root.rglob("*UCSF-PDGM-0004*"):
            if hit.is_dir():
                return hit.parent
        for hit in root.rglob("UCSF-PDGM-metadata.csv"):
            return hit.parent
    else:
        for hit in root.rglob("BT0001"):
            if hit.is_dir():
                return hit.parent
    return list(root.iterdir())[0]


def find_folds(cohort: str) -> Path:
    name = "ucsf_paper_b_folds.csv" if cohort == "ucsf" else "utsw_paper_b_folds.csv"
    for p in Path("/kaggle/input").rglob(name):
        return p
    return REPO_DIR / "data" / "splits" / name


def main() -> None:
    cohort = os.environ.get("COHORT", "ucsf").lower()
    if not REPO_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)])
    else:
        subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
    os.chdir(REPO_DIR)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", "."])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nibabel"])

    out = Path("/kaggle/working") / f"{cohort}_crops"
    cmd = [
        sys.executable,
        "scripts/paper_b_make_crops.py",
        "--cohort",
        cohort,
        "--folds",
        str(find_folds(cohort)),
        "--data-root",
        str(find_data_root(cohort)),
        "--out-dir",
        str(out),
        "--size",
        os.environ.get("SIZE", "96"),
    ]
    max_p = int(os.environ.get("MAX_PATIENTS", "0"))
    if max_p > 0:
        cmd += ["--max-patients", str(max_p)]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
