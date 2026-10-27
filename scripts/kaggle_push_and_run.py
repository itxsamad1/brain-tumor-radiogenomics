#!/usr/bin/env python3
"""Create/update Kaggle datasets and push Paper A training kernel (GPU).

Requires local Kaggle auth (ACCESS_TOKEN). Never prints secrets.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "kaggle_upload" / "btlab-figshare-paper-a"
DS_META = ROOT / "kaggle" / "datasets" / "btlab-figshare-paper-a" / "dataset-metadata.json"
CKPT_DIR = ROOT / "kaggle" / "datasets" / "btlab-paper-a-ckpts"
KERNEL_DIR = ROOT / "kaggle" / "kernels" / "btlab-paper-a-train"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def ensure_dataset_from_export() -> Path:
    if not EXPORT.exists() or not (EXPORT / "splits.csv").exists():
        raise SystemExit(f"Missing export at {EXPORT}. Run scripts/export_figshare_kaggle.py first.")
    # copy metadata into export folder for kaggle CLI
    shutil.copy2(DS_META, EXPORT / "dataset-metadata.json")
    return EXPORT


def dataset_exists(slug: str) -> bool:
    # slug like itxsamad/btlab-figshare-paper-a
    r = subprocess.run(
        ["kaggle", "datasets", "status", slug],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-kernel", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="Push kernel without executing")
    args = parser.parse_args()

    if not args.skip_data:
        export = ensure_dataset_from_export()
        fig_slug = "itxsamad/btlab-figshare-paper-a"
        if dataset_exists(fig_slug):
            run(["kaggle", "datasets", "version", "-p", str(export), "-m", "Paper A PNG+splits update", "-r", "zip"])
        else:
            run(["kaggle", "datasets", "create", "-p", str(export), "-r", "zip"])

        # checkpoint placeholder dataset
        ckpt_slug = "itxsamad/btlab-paper-a-ckpts"
        if dataset_exists(ckpt_slug):
            print(f"Checkpoint dataset exists: {ckpt_slug}")
        else:
            run(["kaggle", "datasets", "create", "-p", str(CKPT_DIR)])

    if not args.skip_kernel:
        push = ["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)]
        if args.no_run:
            push.append("--no-run")
        # Prefer T4 if CLI supports accelerator flag
        try:
            run(push + ["--accelerator", "NvidiaTeslaT4"])
        except subprocess.CalledProcessError:
            print("T4 accelerator flag failed; pushing with metadata enable_gpu only")
            run(push)

    print("Kernel: https://www.kaggle.com/code/itxsamad/btlab-paper-a-train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
