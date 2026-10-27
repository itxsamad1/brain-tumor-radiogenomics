"""Resumable checkpoint helpers for Kaggle 12h sessions and local runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lazy torch import — checkpoints may also be plain JSON meta
    try:
        import torch

        torch.save(state, path)
    except ImportError:
        meta_path = path.with_suffix(".json")
        meta = {k: v for k, v in state.items() if k in {"epoch", "fold", "seed", "best_metric"}}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        raise RuntimeError(
            "PyTorch not installed; wrote meta JSON only. Install torch to save full checkpoints."
        )


def load_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def should_stop(stop_file: Path | str = "STOP") -> bool:
    """Return True if a STOP sentinel file exists (manual pause)."""
    return Path(stop_file).exists()
