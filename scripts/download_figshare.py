#!/usr/bin/env python3
"""Download Jun Cheng Figshare brain tumor dataset (~839 MB) into data/raw/figshare.

Official: https://doi.org/10.6084/m9.figshare.1512427
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


# Figshare article 1512427 — versioned file URLs can change; prefer official page if this 404s.
FIGSHARE_DOWNLOAD = "https://ndownloader.figshare.com/articles/1512427/versions/5"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/figshare"),
        help="Output directory",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "figshare_1512427.zip"

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already present: {dest}")
        return 0

    print(f"Downloading Figshare article 1512427 → {dest}")
    print("If this fails, download manually from https://doi.org/10.6084/m9.figshare.1512427")
    try:
        urllib.request.urlretrieve(FIGSHARE_DOWNLOAD, dest)
    except Exception as e:  # noqa: BLE001
        print(f"Download failed: {e}")
        print("Open the DOI in a browser and save the zip under data/raw/figshare/")
        return 1

    print(f"Done: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
