"""Patient-level registry and frozen split helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def assert_unique_patients(df: pd.DataFrame, patient_col: str = "patient_id") -> None:
    if patient_col not in df.columns:
        raise KeyError(f"Missing column: {patient_col}")
    dupes = df[patient_col][df[patient_col].duplicated()]
    if len(dupes):
        raise ValueError(f"Duplicate patient_id rows: {dupes.unique()[:10].tolist()}")


def hash_split_table(df: pd.DataFrame) -> str:
    """Stable hash of a frozen split table for audit / pre-registration."""
    payload = df.sort_values(list(df.columns)).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_split_manifest(
    splits: pd.DataFrame,
    out_path: Path | str,
    meta: dict[str, Any] | None = None,
) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_path, index=False)
    digest = hash_split_table(splits)
    manifest = {"sha256": digest, "n_rows": len(splits), **(meta or {})}
    out_path.with_suffix(out_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return digest
