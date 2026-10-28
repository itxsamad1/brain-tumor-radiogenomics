#!/usr/bin/env python3
"""Streamlit demo: Paper A ResNet-18 brain-tumor slice classifier + Grad-CAM.

Run from repo root:
  uv run streamlit run app/paper_a_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btlab.data.figshare import CLASS_NAMES  # noqa: E402
from btlab.train.paper_a import IMG_SIZE, build_model  # noqa: E402
from btlab.xai.gradcam import GradCAM, overlay_cam  # noqa: E402

CLASS_LIST = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
DEFAULT_CKPT = ROOT / "experiments" / "paper_a" / "fold1_resnet18" / "checkpoint_best.pt"
SAMPLE_DIR = ROOT / "data" / "kaggle_upload" / "btlab-figshare-paper-a" / "png"


st.set_page_config(
    page_title="BTLab Paper A — MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  .block-container { padding-top: 1.4rem; max-width: 1100px; }
  h1, h2, h3 { font-family: 'IBM Plex Serif', Georgia, serif !important; letter-spacing: -0.02em; }
  .hero {
    background: linear-gradient(135deg, #0f2a3a 0%, #1a4a5c 55%, #2d6a6a 100%);
    color: #f4f7f5; padding: 1.4rem 1.6rem; border-radius: 0;
    margin-bottom: 1.2rem; border-left: 5px solid #c4a35a;
  }
  .hero h1 { color: #f4f7f5 !important; font-size: 1.75rem; margin: 0 0 0.35rem 0; }
  .hero p { margin: 0; opacity: 0.9; font-size: 0.98rem; }
  .metric-strip {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin: 0.8rem 0 1.2rem;
  }
  .metric-card {
    background: #f7f5f0; border: 1px solid #d9d2c5; padding: 0.85rem 1rem;
  }
  .metric-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: #5c6670; }
  .metric-card .value { font-size: 1.35rem; font-weight: 600; color: #0f2a3a; }
  .warn {
    background: #fff6e8; border: 1px solid #e0c090; padding: 0.75rem 1rem; font-size: 0.9rem; color: #4a3b20;
  }
  div[data-testid="stSidebar"] { background: #f3f1eb; }
</style>
""",
    unsafe_allow_html=True,
)


def preprocess(gray: np.ndarray) -> torch.Tensor:
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    x = tf(gray)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x.unsqueeze(0)


def load_gray(file_or_path) -> np.ndarray:
    img = Image.open(file_or_path).convert("L")
    return np.array(img, dtype=np.uint8)


@st.cache_resource(show_spinner="Loading ResNet-18 checkpoint…")
def load_model(ckpt_path: str, device: str):
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    device_t = torch.device(device)
    ckpt = torch.load(path, map_location=device_t, weights_only=False)
    name = ckpt.get("args", {}).get("model", "resnet18")
    model = build_model(name, pretrained=False).to(device_t)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {
        "fold": ckpt.get("fold"),
        "best_f1": ckpt.get("best_f1"),
        "epoch": ckpt.get("epoch"),
        "model": name,
    }
    return model, device_t, meta


def predict_with_cam(model, device, gray: np.ndarray):
    x = preprocess(gray).to(device)
    cam_eng = GradCAM(model, model.layer4)
    cam, pred_idx, probs = cam_eng(x)
    overlay = overlay_cam(np.array(Image.fromarray(gray).resize((IMG_SIZE, IMG_SIZE))), cam)
    return CLASS_LIST[pred_idx], probs, overlay, cam


# --- Hero ---
st.markdown(
    """
<div class="hero">
  <h1>BTLab · Paper A MRI demo</h1>
  <p>Patient-level ResNet-18 · meningioma / glioma / pituitary · Grad-CAM overlay</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="warn">
  <strong>Research demo only — not for clinical use.</strong>
  Trained on Figshare 2D T1c slices. Honest 5-fold macro-F1 ≈ 0.951 (patient-level).
  Slice-level papers often overstate accuracy via leakage.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="metric-strip">
  <div class="metric-card"><div class="label">5-fold macro-F1</div><div class="value">0.951 ± 0.013</div></div>
  <div class="metric-card"><div class="label">Leaky slice F1</div><div class="value">0.978</div></div>
  <div class="metric-card"><div class="label">Tumor-zeroed eval F1</div><div class="value">0.816</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.header("Model")
    ckpt_in = st.text_input("Checkpoint path", value=str(DEFAULT_CKPT))
    use_cpu = st.checkbox("Force CPU", value=not torch.cuda.is_available())
    device_str = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    st.caption(f"Device: `{device_str}`")
    st.divider()
    st.header("Input")
    source = st.radio("Image source", ["Upload", "Sample from dataset"], index=0)
    st.divider()
    st.markdown(
        "Paper package: `experiments/paper_a/`  \n"
        "Manuscript: `Paper_A_Leakage_Shortcuts_Manuscript.docx`"
    )

try:
    model, device, meta = load_model(ckpt_in, device_str)
    best = meta.get("best_f1")
    best_s = f"{best:.3f}" if isinstance(best, (int, float)) else "n/a"
    st.sidebar.success(f"Loaded {meta.get('model')} · fold {meta.get('fold')} · best F1 {best_s}")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

gray = None
label_hint = None

if source == "Upload":
    up = st.file_uploader("Upload MRI slice (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if up is not None:
        gray = load_gray(up)
else:
    samples = []
    if SAMPLE_DIR.exists():
        samples = sorted(SAMPLE_DIR.rglob("*.png"))[:200]
    if not samples:
        st.warning("No sample PNGs found under data/kaggle_upload/btlab-figshare-paper-a/png")
    else:
        pick = st.selectbox(
            "Sample slice",
            samples,
            format_func=lambda p: f"{p.parent.name}/{p.name}",
        )
        gray = load_gray(pick)

if gray is None:
    st.info("Upload a slice or pick a sample to run inference.")
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Input slice")
    st.image(gray, caption=f"{gray.shape[1]}×{gray.shape[0]} grayscale", use_container_width=True)

with st.spinner("Running ResNet-18 + Grad-CAM…"):
    pred_name, probs, overlay, _cam = predict_with_cam(model, device, gray)

with col_b:
    st.subheader("Grad-CAM overlay")
    st.image(overlay, caption=f"Predicted: {pred_name}", use_container_width=True)

st.subheader("Class probabilities")
order = np.argsort(-probs)
prob_cols = st.columns(3)
for i, idx in enumerate(order):
    with prob_cols[i]:
        st.metric(CLASS_LIST[idx].title(), f"{100 * float(probs[idx]):.1f}%")

st.bar_chart({CLASS_LIST[i]: float(probs[i]) for i in range(len(CLASS_LIST))})

with st.expander("About this model"):
    st.markdown(
        """
- **Task:** 3-class tumor-type classification on 2D T1c slices (Figshare Cheng).
- **Training:** patient-level 5-fold; this demo defaults to **fold-1** best checkpoint.
- **Why not 99%?** Slice-level splits leak patients (~99% val overlap). Honest mean macro-F1 ≈ **0.951**.
- **Shortcuts:** Tumor-zeroed training still reaches ~0.957 F1 — background cues remain strong.
        """
    )
