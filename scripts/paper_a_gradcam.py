#!/usr/bin/env python3
"""Generate Grad-CAM panels for Paper A (correct + failure cases)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from btlab.data.figshare import CLASS_NAMES
from btlab.train.paper_a import FigshareSliceDataset, build_model
from btlab.xai.gradcam import save_gradcam_panel


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/paper_a/gradcam"))
    p.add_argument("--n-correct", type=int, default=4)
    p.add_argument("--n-wrong", type=int, default=4)
    p.add_argument("--use-png", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("args", {}).get("model", "resnet18")
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    splits = pd.read_csv(args.splits)
    val = splits[splits["fold"] == args.fold].reset_index(drop=True)
    ds = FigshareSliceDataset(val, args.data_root, train=False, use_png=args.use_png)
    names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

    correct, wrong = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            logits = model(x.unsqueeze(0).to(device))
            pred = int(logits.argmax(dim=1).item())
            (correct if pred == y else wrong).append(i)
            if len(correct) >= 50 and len(wrong) >= 50:
                break

    rng = np.random.default_rng(42)
    pick_c = list(rng.choice(correct, size=min(args.n_correct, len(correct)), replace=False)) if correct else []
    pick_w = list(rng.choice(wrong, size=min(args.n_wrong, len(wrong)), replace=False)) if wrong else []

    meta = []
    for tag, idxs in (("correct", pick_c), ("wrong", pick_w)):
        for j, i in enumerate(idxs):
            x, y = ds[i]
            # reload gray for overlay
            row = val.iloc[i]
            png = args.data_root / "png" / row["subset"] / row["file"]
            gray = np.array(Image.open(png).convert("L").resize((224, 224)))
            info = save_gradcam_panel(
                model,
                x.unsqueeze(0).to(device),
                gray,
                int(y),
                names,
                args.out_dir / f"{tag}_{j}.png",
            )
            info["tag"] = tag
            info["index"] = int(i)
            meta.append(info)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gradcam_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {len(meta)} Grad-CAM panels to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
