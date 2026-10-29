"""Kaggle: exploratory UTSW radiomics baselines (NOT confirmatory external).

Uses full UTSW NIfTI mirror. Per docs/analysis_plan.md this is pipeline validation
until the UCSF-selected model is frozen for a single external run.
"""

from __future__ import annotations

import os

# Reuse the shared radiomics runner with COHORT=utsw
os.environ.setdefault("COHORT", "utsw")

from pathlib import Path
import runpy

# Prefer the sibling script in this repo after clone; for push packaging, inline import path:
ROOT = Path("/kaggle/working/brain-tumor-radiogenomics")
if not (ROOT / "kaggle/kernels/btlab-paper-b-radiomics/btlab-paper-b-radiomics.py").exists():
    # Fall through to cloned runner after git clone inside shared main
    pass

# Execute shared entrypoint module logic by importing after ensuring clone.
import subprocess
import sys

REPO = "https://github.com/itxsamad1/brain-tumor-radiogenomics.git"
if not ROOT.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(ROOT)])
else:
    subprocess.check_call(["git", "-C", str(ROOT), "pull", "--ff-only"])

sys.path.insert(0, str(ROOT / "kaggle" / "kernels" / "btlab-paper-b-radiomics"))
os.chdir(ROOT)
runpy.run_path(str(ROOT / "kaggle/kernels/btlab-paper-b-radiomics/btlab-paper-b-radiomics.py"), run_name="__main__")
