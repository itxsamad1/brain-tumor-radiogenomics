# Paper B — Progress Report

**Status:** In progress — fold-1 imaging baseline done; multi-fold + clinical fusion running on Kaggle  
**Question:** Calibrated, overlap-safe IDH / 1p19q / grade prediction across cohorts

## Must-beat clinical baselines (age + sex LR, 5-fold)

| Cohort | Role | n (IDH labeled) | AUROC (mean ± std) | Bal. acc |
| --- | --- | ---: | ---: | ---: |
| UTSW-Glioma | external (planned) | 622 | **0.875 ± 0.024** | 0.809 |
| UCSF-PDGM | development | 501 (103 mut / 398 wt) | **0.904 ± 0.025** | 0.850 |

UCSF: **298/501** patients carry BraTS21 IDs → overlap exclusions required before any BraTS-pretrained external claim.

## Imaging 2.5D ResNet-18 (UTSW 2D PNG mirror, patient-level)

| Fold | Best AUROC | AUPRC | Bal. acc | vs age (0.875) |
| --- | ---: | ---: | ---: | --- |
| 1 (imaging-only) | **0.870** | 0.714 | 0.777 | −0.005 (does **not** beat age) |
| 2–5 + clinical fusion | running on Kaggle | — | — | https://www.kaggle.com/code/itxsamad/btlab-paper-b-idh |

Fold-1 artifacts: `fold1_idh_2p5d/`

## Interpretation so far

Pure mid-volume 2.5D MRI is **not** enough on UTSW — age already carries most IDH signal. Next models must either (a) fuse clinical covariates, (b) use tumor-centered / 3D features, or (c) show value via calibration / external generalization beyond AUROC alone.

## Done

| Item | Path |
| --- | --- |
| Official UTSW TSV (625 pts) | `data/raw/utsw-glioma/` |
| UTSW frozen folds | `data/splits/utsw_paper_b_folds.csv` |
| UCSF labels + BraTS flags | `data/splits/ucsf_paper_b_*.csv` |
| Age±sex LR (both cohorts) | `experiments/paper_b/age_baseline/`, `ucsf_age_baseline/` |
| Kaggle kernel | https://www.kaggle.com/code/itxsamad/btlab-paper-b-idh |

## Next

1. Finish 5-fold imaging±clinical fusion on Kaggle  
2. Radiomics + TabPFN/XGBoost baselines (CPU)  
3. Official Aspera NIfTI + UCSF-PDGM imaging packaging  
4. Lock `docs/analysis_plan.md` before external peeking  
