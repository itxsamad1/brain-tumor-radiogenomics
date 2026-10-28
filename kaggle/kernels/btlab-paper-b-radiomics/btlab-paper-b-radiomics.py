"""Kaggle: Paper B radiomics must-beat baselines (UCSF development by default).

Env:
  COHORT=ucsf|utsw
  MAX_PATIENTS=0   # 0=all available on the mounted dataset
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
    if cohort == "ucsf":
        for hit in root.rglob("*UCSF-PDGM-0004*"):
            # expect .../UCSF-PDGM/UCSF-PDGM-0004_nifti → cohort root = parent of patient dir's parent or parent
            if hit.is_dir():
                # prefer directory that contains many patient folders
                parent = hit.parent
                if parent.name.lower().startswith("ucsf") or any(parent.glob("*UCSF-PDGM*")):
                    return parent
                return parent
        for hit in root.rglob("UCSF-PDGM-metadata.csv"):
            return hit.parent
    else:
        for hit in root.rglob("BT0001"):
            if hit.is_dir():
                return hit.parent
        for hit in root.rglob("UTSW-Glioma"):
            if hit.is_dir():
                return hit
    roots = list(root.iterdir())
    if not roots:
        raise FileNotFoundError("No /kaggle/input datasets")
    return roots[0]


def find_folds(cohort: str) -> Path:
    name = "ucsf_paper_b_folds.csv" if cohort == "ucsf" else "utsw_paper_b_folds.csv"
    for p in Path("/kaggle/input").rglob(name):
        return p
    local = REPO_DIR / "data" / "splits" / name
    if local.exists():
        return local
    raise FileNotFoundError(name)


def main() -> None:
    cohort = os.environ.get("COHORT", "ucsf").lower()
    if not REPO_DIR.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)])
    else:
        subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
    os.chdir(REPO_DIR)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".[medical]"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nibabel", "xgboost"])

    data_root = find_data_root(cohort)
    folds = find_folds(cohort)
    out = Path("/kaggle/working") / f"{cohort}_radiomics"
    out.mkdir(parents=True, exist_ok=True)
    print("cohort", cohort, "data_root", data_root, "folds", folds)

    cmd = [
        sys.executable,
        "scripts/paper_b_radiomics_baseline.py",
        "--cohort",
        cohort,
        "--folds",
        str(folds),
        "--data-root",
        str(data_root),
        "--out-dir",
        str(out),
        "--features-cache",
        str(out / "features.csv"),
    ]
    max_p = int(os.environ.get("MAX_PATIENTS", "0"))
    if max_p > 0:
        cmd += ["--max-patients", str(max_p)]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Done", out)


if __name__ == "__main__":
    main()
