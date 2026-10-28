# Paper B — Progress Report

**Status:** UTSW 5-fold 2.5D IDH run complete; clinical baselines on UTSW + UCSF ready  
**Question:** Calibrated, overlap-safe IDH / 1p19q / grade prediction across cohorts

## Must-beat clinical baselines (age + sex LR, 5-fold)

| Cohort | Role | n (IDH labeled) | AUROC (mean ± std) | Bal. acc |
| --- | --- | ---: | ---: | ---: |
| UTSW-Glioma | external (planned) | 622 | **0.875 ± 0.024** | 0.809 |
| UCSF-PDGM | development | 501 (103 mut / 398 wt) | **0.904 ± 0.025** | 0.850 |

UCSF: **298/501** patients carry BraTS21 IDs → overlap exclusions required before any BraTS-pretrained external claim.

## Imaging 2.5D ResNet-18 (UTSW 2D PNG mirror, patient-level)

| Fold | Best AUROC | Notes |
| --- | ---: | --- |
| 1 | **0.870** | imaging-only |
| 2 | **0.892** | multi-fold Kaggle (clinical fusion enabled in script) |
| 3 | **0.859** | early-stopped |
| 4 | **0.804** | weakest fold |
| 5 | **0.843** | early-stopped |
| **Mean ± SD** | **0.854 ± 0.030** | **does not beat age LR (0.875)** |

Artifacts: `fold{1..5}_idh_2p5d/`, aggregate `idh_2p5d_summary.json`.

## Interpretation

Mid-slice 2.5D ResNet on UTSW is **below** a trivial age+sex logistic model. Paper B’s value cannot be “higher AUROC than age” with this weak imaging recipe — need tumor-centered 3D/radiomics, stronger fusion, and/or claims on calibration / external transport / incremental value (ΔAUROC + likelihood ratio).

## Done

| Item | Path |
| --- | --- |
| UTSW + UCSF labels/folds | `data/splits/` |
| Age±sex LR both cohorts | `age_baseline/`, `ucsf_age_baseline/` |
| Kaggle IDH kernel | https://www.kaggle.com/code/itxsamad/btlab-paper-b-idh |

## Next

1. Tumor-centered crops / radiomics + XGBoost/TabPFN (must-beat table)  
2. UCSF-PDGM imaging packaging for development CV  
3. Lock `docs/analysis_plan.md` before external peeking  
4. Optional: re-run fold-1 with same fusion settings for strict comparability  
