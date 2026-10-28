"""Lightweight multi-modal radiomics from NIfTI image + tumor mask.

Uses nibabel + numpy only (no pyradiomics required) so Kaggle installs stay reliable.
Features: shape + first-order intensity per modality and per mask compartment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    raise ImportError("nibabel is required: uv pip install nibabel") from e


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    aff = img.affine
    return np.squeeze(data), aff


def _voxel_volume_mm3(affine: np.ndarray) -> float:
    spacing = np.abs(np.diag(affine)[:3])
    return float(np.prod(spacing))


def shape_features(mask: np.ndarray, affine: np.ndarray, prefix: str = "shape") -> dict[str, float]:
    m = mask > 0
    n = int(m.sum())
    vol = n * _voxel_volume_mm3(affine)
    if n == 0:
        return {
            f"{prefix}_voxels": 0.0,
            f"{prefix}_volume_mm3": 0.0,
            f"{prefix}_surface_approx": 0.0,
            f"{prefix}_sphericity_proxy": 0.0,
            f"{prefix}_bbox_x": 0.0,
            f"{prefix}_bbox_y": 0.0,
            f"{prefix}_bbox_z": 0.0,
        }
    # approximate surface via face differences
    surf = 0
    for ax in range(3):
        diff = np.diff(m.astype(np.int8), axis=ax)
        surf += int(np.abs(diff).sum())
    # bbox extents
    coords = np.argwhere(m)
    extents = (coords.max(axis=0) - coords.min(axis=0) + 1).astype(float)
    # sphericity proxy: 1 when sphere-like
    radius = (3 * vol / (4 * np.pi)) ** (1 / 3) if vol > 0 else 0.0
    sphere_surf = 4 * np.pi * radius * radius
    sph = float(sphere_surf / (surf * (_voxel_volume_mm3(affine) ** (2 / 3)) + 1e-8))
    return {
        f"{prefix}_voxels": float(n),
        f"{prefix}_volume_mm3": float(vol),
        f"{prefix}_surface_approx": float(surf),
        f"{prefix}_sphericity_proxy": sph,
        f"{prefix}_bbox_x": float(extents[0]),
        f"{prefix}_bbox_y": float(extents[1]),
        f"{prefix}_bbox_z": float(extents[2]),
    }


def first_order(img: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    vals = img[mask > 0].astype(np.float64)
    if vals.size == 0:
        keys = ["mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90", "skew", "kurt", "energy", "entropy"]
        return {f"{prefix}_{k}": 0.0 for k in keys}
    # robust normalize within ROI for entropy bins
    hist, _ = np.histogram(vals, bins=32, density=True)
    hist = hist + 1e-12
    hist = hist / hist.sum()
    entropy = float(-(hist * np.log2(hist)).sum())
    centered = vals - vals.mean()
    std = float(vals.std() + 1e-12)
    skew = float(np.mean(centered**3) / (std**3))
    kurt = float(np.mean(centered**4) / (std**4) - 3.0)
    qs = np.percentile(vals, [10, 25, 50, 75, 90])
    return {
        f"{prefix}_mean": float(vals.mean()),
        f"{prefix}_std": float(vals.std()),
        f"{prefix}_min": float(vals.min()),
        f"{prefix}_max": float(vals.max()),
        f"{prefix}_p10": float(qs[0]),
        f"{prefix}_p25": float(qs[1]),
        f"{prefix}_p50": float(qs[2]),
        f"{prefix}_p75": float(qs[3]),
        f"{prefix}_p90": float(qs[4]),
        f"{prefix}_skew": skew,
        f"{prefix}_kurt": kurt,
        f"{prefix}_energy": float(np.sum(vals.astype(np.float64) ** 2)),
        f"{prefix}_entropy": entropy,
    }


def extract_case_features(
    images: dict[str, Path],
    mask_path: Path,
    compartments: Iterable[int] | None = None,
) -> dict[str, float]:
    """Extract features for one patient.

    images: modality_name -> nifti path (e.g. t1c, flair, t2, t1)
    mask: integer segmentation (BraTS-style 1/2/4 or binary)
    """
    mask, aff = load_nifti(mask_path)
    mask = np.rint(mask).astype(np.int16)
    feats: dict[str, float] = {}
    # whole tumor = any positive label
    wt = (mask > 0).astype(np.uint8)
    feats.update(shape_features(wt, aff, prefix="wt_shape"))

    if compartments is None:
        # common BraTS: 1 NCR, 2 ED, 4 ET — also accept binary
        comps = {
            "wt": wt,
            "et": ((mask == 4) | (mask == 1)).astype(np.uint8) if (mask >= 4).any() else wt,
            "ed": (mask == 2).astype(np.uint8) if (mask == 2).any() else np.zeros_like(wt),
        }
    else:
        comps = {f"c{c}": (mask == c).astype(np.uint8) for c in compartments}
        comps["wt"] = wt

    for mod, path in images.items():
        if path is None or not Path(path).exists():
            continue
        img, _ = load_nifti(Path(path))
        if img.shape != mask.shape:
            # skip mismatched geometry rather than silently wrong features
            feats[f"{mod}_shape_mismatch"] = 1.0
            continue
        feats[f"{mod}_shape_mismatch"] = 0.0
        # z-score inside brain-ish nonzero
        brain = img[np.isfinite(img) & (np.abs(img) > 0)]
        if brain.size > 10:
            mu, sd = float(brain.mean()), float(brain.std() + 1e-8)
            img_n = (img - mu) / sd
        else:
            img_n = img.astype(np.float64)
        for cname, cm in comps.items():
            if cm.sum() == 0:
                continue
            feats.update(first_order(img_n, cm, prefix=f"{mod}_{cname}"))
    return feats


def tumor_centered_crop(
    images: dict[str, np.ndarray],
    mask: np.ndarray,
    size: tuple[int, int, int] = (96, 96, 96),
) -> dict[str, np.ndarray]:
    """Return float32 crops centered on tumor centroid; pad if needed."""
    m = mask > 0
    if not m.any():
        center = np.array(mask.shape) // 2
    else:
        center = np.argwhere(m).mean(axis=0).astype(int)
    out = {}
    for name, vol in images.items():
        crop = np.zeros(size, dtype=np.float32)
        half = np.array(size) // 2
        src_start = np.maximum(center - half, 0)
        src_end = np.minimum(center - half + np.array(size), np.array(vol.shape))
        dst_start = half - (center - src_start)
        dst_end = dst_start + (src_end - src_start)
        sl_src = tuple(slice(int(a), int(b)) for a, b in zip(src_start, src_end))
        sl_dst = tuple(slice(int(a), int(b)) for a, b in zip(dst_start, dst_end))
        crop[sl_dst] = vol[sl_src].astype(np.float32)
        out[name] = crop
    return out
