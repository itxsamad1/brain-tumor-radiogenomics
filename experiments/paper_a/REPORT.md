# Paper A — Final Experimental Report

**Title angle:** Reality check — leakage and shortcuts in 2D brain-tumor MRI classification  
**Status:** Core experiments complete (5-fold patient-level + leaky baseline + Grad-CAM + tumor-masked control)  
**Plan:** [`docs/PAPER_PUBLISH_PLAN.md`](../../docs/PAPER_PUBLISH_PLAN.md)

## Headline results

| Protocol | Macro-F1 | Accuracy | Notes |
| --- | ---: | ---: | --- |
| Patient-level 5-fold ResNet-18 (mean ± std) | **0.951 ± 0.013** | **0.956 ± 0.013** | Primary claim |
| Slice-level leaky split (same model) | **0.978** | **0.980** | Inflated by patient overlap |
| Tumor-zeroed eval (fold-1 ckpt) | **0.816** | **0.856** | Drop vs 0.962 → uses tumor region |
| Tumor-zeroed train-from-scratch (fold 1, 10 ep) | **0.957** | — | ≈ patient-level ⇒ strong non-tumor shortcuts |

Leakage audit: **99.0%** of val slices share a patient with the train set under slice-level random split (`leaky_slice_split/leakage_stats.json`).

## Patient-level 5-fold (ResNet-18, ImageNet init, 30 epochs)

| Fold | Val patients | Accuracy | Macro-F1 | AUROC (OVR) | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 45 | 0.967 | **0.962** | 0.998 | 0.017 |
| 2 | 46 | 0.944 | **0.940** | 0.993 | 0.039 |
| 3 | — | 0.956 | **0.952** | 0.992 | 0.031 |
| 4 | — | 0.975 | **0.969** | 0.996 | 0.021 |
| 5 | — | 0.939 | **0.933** | 0.989 | 0.043 |

Per-fold packages: `fold{1..5}_resnet18/` (metrics, CIs, confusion, ROC/PR, reliability, training curves).

## Audits

| Audit | Result | Path |
| --- | --- | --- |
| dHash near-duplicates | 5 cross-patient identical-hash groups | `audits/` |
| Slice-level leaky split | F1 0.978 vs patient mean 0.951; 199 patients in both splits | `leaky_slice_split/` |
| Grad-CAM | 8 panels (correct + wrong) | `gradcam/` |
| Tumor-masked control | Eval F1 drop 0.962→0.816; train-on-masked still learns well | `tumor_masked/` |

## Interpretation (for the paper)

1. **Patient-level splits are necessary but not sufficient marketing:** honest 5-fold macro-F1 ≈ 0.95, not 0.99.
2. **Slice-level papers inflate scores** via near-total patient overlap (~99% of val slices).
3. **Models do look at tumor** (zeroing tumor hurts a pretrained net) **but non-tumor cues remain strong** (training with tumor zeroed still recovers high F1) — classic shortcut / context leakage story for Figshare-style 2D sets.

## Figures

- `figures/patient_vs_slice_macro_f1.png` — main comparison bar (after `scripts/paper_a_aggregate.py`)
- Per-fold: `fold*_resnet18/figures/`
- Grad-CAM: `gradcam/*.png`
- Leaky: `leaky_slice_split/figures/`

## Reproducibility

- Data: Figshare Cheng DOI `10.6084/m9.figshare.1512427`; Kaggle pack `itxsamad/btlab-figshare-paper-a`
- Splits: `data/kaggle_upload/btlab-figshare-paper-a/splits.csv` (patient-majority from `cvind`)
- Kaggle train: https://www.kaggle.com/code/itxsamad/btlab-paper-a-full
- Resume bug fix (fold isolation): commit `cfd68ea`

## Deliverables

- Manuscript (Word): `Paper_A_Leakage_Shortcuts_Manuscript.docx`
- Demo UI: `uv run streamlit run app/paper_a_demo.py`

## Optional remaining

- [ ] Optional ResNet-50 / second seed
- [ ] Expanded systematic literature table
- [ ] Sync final weights to `itxsamad/btlab-paper-a-ckpts`
- [ ] Camera-ready journal formatting
