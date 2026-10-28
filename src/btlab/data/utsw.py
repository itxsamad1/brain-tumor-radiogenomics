"""UTSW-Glioma (TCIA) label helpers for Paper B.

Cite: https://doi.org/10.7937/DFAE-1B86
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_utsw_metadata(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    df = df.rename(
        columns={
            "Subject ID": "patient_id",
            "Sex at birth": "sex",
            "Age at Imaging": "age",
            "Tumor Type": "tumor_type_raw",
            "Tumor Grade": "grade",
            "IDH": "idh_raw",
            "1p19Q CODEL": "codel_raw",
            "MGMT": "mgmt_raw",
            "Scanner Strength": "field_strength",
            "Scanner Make": "scanner_make",
        }
    )
    return df


def normalize_idh(series: pd.Series) -> pd.Series:
    m = {
        "wild type": 0,
        "wildtype": 0,
        "wt": 0,
        "mutated": 1,
        "mutant": 1,
        "idh-mutant": 1,
        "idh-wildtype": 0,
    }
    out = series.astype(str).str.strip().str.lower().map(m)
    return out


def normalize_codel(series: pd.Series) -> pd.Series:
    m = {
        "co-deleted": 1,
        "codeleted": 1,
        "non co-deleted": 0,
        "non-co-deleted": 0,
        "noncodeleted": 0,
    }
    return series.astype(str).str.strip().str.lower().map(m)


def build_paper_b_labels(meta: pd.DataFrame) -> pd.DataFrame:
    df = meta.copy()
    df["idh"] = normalize_idh(df["idh_raw"])
    df["codel"] = normalize_codel(df["codel_raw"])
    df["sex_male"] = (df["sex"].astype(str).str.upper().str.startswith("M")).astype(int)
    # 1p/19q task only meaningful within IDH-mutant
    df["codel_in_idhmut"] = np.where(df["idh"] == 1, df["codel"], np.nan)
    keep = [
        "patient_id",
        "age",
        "sex",
        "sex_male",
        "grade",
        "tumor_type_raw",
        "idh",
        "codel",
        "codel_in_idhmut",
        "mgmt_raw",
        "field_strength",
        "scanner_make",
    ]
    return df[[c for c in keep if c in df.columns]]


def patient_stratified_folds(
    labels: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    strata_col: str = "idh",
) -> pd.DataFrame:
    """Assign each patient to a fold; stratify by binary strata when possible."""
    from sklearn.model_selection import StratifiedKFold

    df = labels.dropna(subset=[strata_col]).copy()
    y = df[strata_col].astype(int).to_numpy()
    ids = df["patient_id"].to_numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), dtype=int)
    for i, (_, val_idx) in enumerate(skf.split(ids, y), start=1):
        fold[val_idx] = i
    out = df.copy()
    out["fold"] = fold
    return out
