"""BraTS ↔ TCIA overlap helpers for leakage-safe Paper B claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_ucsf_overlap_registry(labels_csv: Path | str, out_json: Path | str) -> dict:
    """Flag UCSF patients that appear in BraTS 2021 via official metadata columns."""
    df = pd.read_csv(labels_csv)
    if "in_brats21" not in df.columns:
        df["in_brats21"] = df.get("brats21_id", pd.Series([None] * len(df))).notna()
        df["in_brats21"] = df["in_brats21"] & (df.get("brats21_id", "").astype(str).str.len() > 0)

    overlap_ids = sorted(df.loc[df["in_brats21"], "patient_id"].astype(str).tolist())
    non_overlap = sorted(df.loc[~df["in_brats21"].astype(bool), "patient_id"].astype(str).tolist())
    reg = {
        "cohort": "UCSF-PDGM",
        "n_total": int(len(df)),
        "n_in_brats21": int(len(overlap_ids)),
        "n_non_brats21": int(len(non_overlap)),
        "brats21_patient_ids": overlap_ids,
        "non_brats21_patient_ids": non_overlap,
        "rule": (
            "Patients with a BraTS21 ID must not be used as 'external' evidence for any model "
            "that saw BraTS 2021 (or BraTS-overlapping UCSF cases) during training/pretraining."
        ),
        "source_file": str(labels_csv),
        "source_sha256": file_sha256(Path(labels_csv)) if Path(labels_csv).exists() else None,
    }
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg


def assert_no_patient_leak(train_ids, test_ids) -> None:
    leak = set(map(str, train_ids)) & set(map(str, test_ids))
    if leak:
        raise AssertionError(f"Patient leakage detected ({len(leak)} ids), e.g. {sorted(leak)[:5]}")
