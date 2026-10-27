"""Jun Cheng Figshare 2D brain tumor dataset helpers (Paper A).

Original: https://doi.org/10.6084/m9.figshare.1512427
Labels: 1=meningioma, 2=glioma, 3=pituitary (per dataset README).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASS_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def _load_mat(path: Path) -> dict[str, Any]:
    """Load MATLAB .mat (v5 via scipy, v7.3 HDF5 via h5py)."""
    try:
        from scipy.io import loadmat

        return loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import h5py
        import numpy as np

        out: dict[str, Any] = {}
        with h5py.File(path, "r") as f:
            # Typical layout: cjdata/{label,PID,image,...}
            if "cjdata" in f:
                g = f["cjdata"]

                class Obj:
                    pass

                obj = Obj()
                if "label" in g:
                    obj.label = int(np.array(g["label"]).squeeze())
                if "PID" in g:
                    # PID stored as UTF-16 code units in HDF5 matlab files
                    raw = np.array(g["PID"]).squeeze()
                    if raw.dtype.kind in "iu":
                        obj.PID = "".join(chr(int(c)) for c in raw.flatten())
                    else:
                        obj.PID = str(raw)
                if "image" in g:
                    obj.image = np.array(g["image"])
                if "tumorMask" in g:
                    obj.tumorMask = np.array(g["tumorMask"])
                out["cjdata"] = obj
            else:
                for k in f.keys():
                    out[k] = np.array(f[k])
        return out


def parse_patient_table(mat_dir: Path | str) -> pd.DataFrame:
    """Build a patient/slice table from extracted Figshare .mat files.

    Each .mat typically contains fields: cjdata.image, cjdata.label, cjdata.PID,
    cjdata.tumorBorder, cjdata.tumorMask (field names vary slightly by version).
    """
    mat_dir = Path(mat_dir)
    rows: list[dict[str, Any]] = []
    files = sorted(mat_dir.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files under {mat_dir}")

    for fp in files:
        if fp.name.lower() == "cvind.mat":
            continue
        data = _load_mat(fp)
        # Prefer nested cjdata struct if present
        obj = data.get("cjdata", data)
        label = int(getattr(obj, "label", obj.get("label") if isinstance(obj, dict) else None))
        pid = str(getattr(obj, "PID", getattr(obj, "pid", "unknown")))
        rows.append(
            {
                "path": str(fp),
                "patient_id": pid,
                "label": label,
                "class_name": CLASS_NAMES.get(label, f"class_{label}"),
            }
        )
    return pd.DataFrame(rows)


def load_cvind(path: Path | str) -> np.ndarray:
    """Load official patient-wise fold indices from cvind.mat if present."""
    data = _load_mat(Path(path))
    key = "cvind" if "cvind" in data else next(k for k in data if not k.startswith("__"))
    return np.asarray(data[key]).squeeze()


def patient_level_split(
    table: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test by unique patient_id (never by slice)."""
    rng = np.random.default_rng(seed)
    patients = table[["patient_id", "label"]].drop_duplicates("patient_id")
    # Stratify roughly by majority label per patient
    labels = patients["label"].to_numpy()
    ids = patients["patient_id"].to_numpy()
    order = rng.permutation(len(ids))
    ids, labels = ids[order], labels[order]

    n = len(ids)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test : n_test + n_val])
    train_ids = set(ids[n_test + n_val :])

    def role(pid: str) -> str:
        if pid in test_ids:
            return "test"
        if pid in val_ids:
            return "val"
        if pid in train_ids:
            return "train"
        return "train"

    out = table.copy()
    out["split"] = out["patient_id"].map(role)
    return out
