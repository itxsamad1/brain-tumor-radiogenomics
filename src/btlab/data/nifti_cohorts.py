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


def _ucsf_id_variants(patient_id: str) -> list[str]:
    """Metadata uses UCSF-PDGM-004; Kaggle folders often use UCSF-PDGM-0004."""
    pid = str(patient_id).strip()
    out = [pid]
    if "-" in pid:
        prefix, _, suffix = pid.rpartition("-")
        if suffix.isdigit():
            n = int(suffix)
            for width in (3, 4, 5):
                out.append(f"{prefix}-{n:0{width}d}")
    # unique, preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def find_ucsf_patient_dir(root: Path, patient_id: str) -> Path | None:
    # layouts seen on Kaggle mirrors (zero-padding differs from metadata IDs)
    for pid in _ucsf_id_variants(patient_id):
        for cand in [
            root / "UCSF-PDGM" / f"{pid}_nifti",
            root / "ucsf" / f"{pid}_nifti",
            root / f"{pid}_nifti",
            root / pid,
        ]:
            if cand.is_dir():
                return cand
    for pid in _ucsf_id_variants(patient_id):
        hits = [p for p in root.rglob(f"*{pid}*") if p.is_dir()]
        for h in hits:
            if any(p.is_file() and ".nii" in p.name.lower() for p in h.rglob("*")):
                return h
    return None


def _first_nifti_file(
    root: Path,
    patterns: list[str],
    *,
    exclude_substrings: list[str] | None = None,
) -> Path | None:
    """Kaggle UCSF mirrors often nest files as ``name.nii/name.nii`` directories."""
    exclude = [e.lower() for e in (exclude_substrings or [])]

    def _ok(name: str) -> bool:
        low = name.lower()
        if ".nii" not in low:
            return False
        return not any(e in low for e in exclude)

    for pat in patterns:
        for hit in root.rglob(pat):
            if hit.is_file() and _ok(hit.name):
                return hit
            if hit.is_dir() and _ok(hit.name):
                nested = [p for p in hit.rglob("*") if p.is_file() and _ok(p.name)]
                if nested:
                    return nested[0]
    return None


def resolve_ucsf_case(root: Path, patient_id: str) -> dict[str, Path] | None:
    pdir = find_ucsf_patient_dir(root, patient_id)
    if pdir is None:
        return None

    images = {
        "t1c": _first_nifti_file(pdir, ["*T1c*", "*T1gad*", "*t1ce*"]),
        "flair": _first_nifti_file(pdir, ["*FLAIR*", "*flair*"]),
        "t2": _first_nifti_file(pdir, ["*T2*", "*t2*"], exclude_substrings=["t1", "flair"]),
        "t1": _first_nifti_file(
            pdir,
            ["*T1_bias*", "*T1w*", "*T1n*", "*T1*"],
            exclude_substrings=["t1c", "t1gad", "t1ce", "flair", "t2"],
        ),
    }
    mask = _first_nifti_file(pdir, ["*tumor_segmentation*", "*seg*"])
    if mask is None or images["t1c"] is None:
        return None
    out = {k: v for k, v in images.items() if v is not None}
    out["mask"] = mask
    return out
