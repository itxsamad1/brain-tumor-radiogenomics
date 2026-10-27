#!/usr/bin/env python3
"""After a Kaggle kernel finishes: pull output checkpoints and version ckpt dataset.

Usage:
  uv run python scripts/kaggle_sync_checkpoints.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "kaggle_upload" / "paper_a_ckpt_staging"
CKPT_DS = ROOT / "kaggle" / "datasets" / "btlab-paper-a-ckpts"
KERNEL = "itxsamad/btlab-paper-a-train"


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    subprocess.check_call(["kaggle", "kernels", "output", KERNEL, "-p", str(OUT)])

    # copy pt/json into ckpt dataset folder
    for name in ("checkpoint_last.pt", "checkpoint_best.pt", "status.json", "history.json"):
        hits = list(OUT.rglob(name))
        if hits:
            shutil.copy2(hits[0], CKPT_DS / name)
            print("copied", hits[0], "->", CKPT_DS / name)

    if not (CKPT_DS / "checkpoint_last.pt").exists():
        print("No checkpoint_last.pt in kernel output yet.")
        return 1

    subprocess.check_call(
        ["kaggle", "datasets", "version", "-p", str(CKPT_DS), "-m", "Resume checkpoint from kernel output"]
    )
    print("Updated itxsamad/btlab-paper-a-ckpts — next kernel run will auto-resume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
