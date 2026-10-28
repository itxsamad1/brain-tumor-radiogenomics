"""Resolve multi-contrast NIfTI + mask paths for UTSW / UCSF-PDGM layouts."""

from __future__ import annotations

from pathlib import Path


def find_utsw_patient_dir(root: Path, patient_id: str) -> Path | None:
    for cand in [
        root / "UTSW-Glioma" / patient_id,
        root / patient_id,
        root / "UTSW_Glioma" / patient_id,
    ]:
        if cand.is_dir():
            return cand
    hits = list(root.rglob(patient_id))
    for h in hits:
        if h.is_dir() and any(h.glob("*.nii*")):
            return h
    return None


def resolve_utsw_case(root: Path, patient_id: str) -> dict[str, Path] | None:
    pdir = find_utsw_patient_dir(root, patient_id)
    if pdir is None:
        return None
    # Prefer ANTs skull-stripped when present; fallback to non-ants
    def pick(*names: str) -> Path | None:
        for n in names:
            for ext in (".nii", ".nii.gz"):
                fp = pdir / f"{n}{ext}"
                if fp.exists():
                    return fp
        return None

    images = {
        "t1c": pick("brain_t1ce_ants", "brain_t1ce"),
        "flair": pick("brain_fl_ants", "brain_flair", "brain_fl"),
        "t2": pick("brain_t2_ants", "brain_t2"),
        "t1": pick("brain_t1_ants", "brain_t1"),
    }
    mask = pick(
        "tumorseg_manual_correction",
        "rtumorseg_manual_correction",
        "tumorseg_FeTS",
    )
    if mask is None or images["t1c"] is None:
        return None
    out = {k: v for k, v in images.items() if v is not None}
    out["mask"] = mask
    return out


def find_ucsf_patient_dir(root: Path, patient_id: str) -> Path | None:
    # layouts seen on Kaggle mirrors
    for cand in [
        root / "UCSF-PDGM" / f"{patient_id}_nifti",
        root / f"{patient_id}_nifti",
        root / patient_id,
    ]:
        if cand.is_dir():
            return cand
    hits = [p for p in root.rglob(f"*{patient_id}*") if p.is_dir()]
    for h in hits:
        if any(h.rglob("*.nii*")):
            return h
    return None


def resolve_ucsf_case(root: Path, patient_id: str) -> dict[str, Path] | None:
    pdir = find_ucsf_patient_dir(root, patient_id)
    if pdir is None:
        return None

    def find_one(patterns: list[str]) -> Path | None:
        for pat in patterns:
            hits = list(pdir.rglob(pat))
            if hits:
                return hits[0]
        return None

    images = {
        "t1c": find_one(["*T1c*.nii*", "*T1gad*.nii*", "*t1ce*.nii*"]),
        "flair": find_one(["*FLAIR*.nii*", "*flair*.nii*"]),
        "t2": find_one(["*T2*.nii*", "*t2*.nii*"]),
        "t1": find_one(["*T1_bias*.nii*", "*T1w*.nii*", "*t1.nii*"]),
    }
    mask = find_one(["*tumor_segmentation*.nii*", "*seg*.nii*"])
    if mask is None or images["t1c"] is None:
        return None
    out = {k: v for k, v in images.items() if v is not None}
    out["mask"] = mask
    return out
