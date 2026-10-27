"""Grad-CAM for Paper A ResNet backbones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, o):
        self.activations = o.detach()

    def _bwd(self, _m, _gi, go):
        self.gradients = go[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, torch.softmax(logits, dim=1)[0].detach().cpu().numpy()


def overlay_cam(gray_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    import matplotlib.cm as cm

    base = np.stack([gray_uint8] * 3, axis=-1).astype(np.float32) / 255.0
    heat = cm.jet(cam)[..., :3]
    out = (1 - alpha) * base + alpha * heat
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)


def save_gradcam_panel(
    model: torch.nn.Module,
    x: torch.Tensor,
    gray: np.ndarray,
    true_label: int,
    class_names: list[str],
    out_path: Path,
) -> dict:
    # ResNet: layer4
    layer = model.layer4
    cam_eng = GradCAM(model, layer)
    cam, pred, probs = cam_eng(x)
    overlay = overlay_cam(gray, cam)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # side-by-side
    g = Image.fromarray(gray).convert("RGB").resize(overlay.size)
    w, h = overlay.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(g, (0, 0))
    canvas.paste(overlay, (w, 0))
    canvas.save(out_path)
    return {
        "true": class_names[true_label],
        "pred": class_names[pred],
        "probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "path": str(out_path),
    }
