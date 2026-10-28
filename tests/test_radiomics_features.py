"""Unit tests for lightweight radiomics (no external NIfTI required)."""

from __future__ import annotations

import numpy as np

from btlab.radiomics.features import first_order, shape_features, tumor_centered_crop


def test_shape_and_first_order_nonzero():
    aff = np.eye(4)
    aff[0, 0] = aff[1, 1] = aff[2, 2] = 1.0
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[10:20, 10:20, 10:20] = 1
    img = np.random.randn(32, 32, 32).astype(np.float32)
    s = shape_features(mask, aff)
    assert s["wt_shape_voxels"] == 1000
    f = first_order(img, mask, prefix="t1c_wt")
    assert "t1c_wt_mean" in f
    assert np.isfinite(f["t1c_wt_mean"])


def test_tumor_centered_crop_shape():
    mask = np.zeros((40, 40, 40), dtype=np.uint8)
    mask[18:22, 18:22, 18:22] = 1
    imgs = {"t1c": np.random.randn(40, 40, 40).astype(np.float32)}
    crops = tumor_centered_crop(imgs, mask, size=(16, 16, 16))
    assert crops["t1c"].shape == (16, 16, 16)
