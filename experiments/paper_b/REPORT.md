# Paper B — Progress Report

**Status:** UCSF development radiomics must-beat table **complete**; crops + UTSW exploratory radiomics still running  
**Plan:** [`docs/analysis_plan.md`](../../docs/analysis_plan.md) (locked confirmatory protocol)

## Scientific roles (correct)

| Role | Cohort | Status |
| --- | --- | --- |
| Development CV | **UCSF-PDGM** | Radiomics table done (495 pts) |
| Primary external (once) | **UTSW-Glioma** | Labels ready; confirmatory run **not** started |
| Exploratory only | UTSW 2.5D PNG ResNet | Mean AUROC 0.854 < age 0.875 |

## UCSF development — IDH must-beat table (patient-level 5-fold)

Source: `ucsf_radiomics_summary.json` / `ucsf_radiomics_features.csv` (87 features; T1c+FLAIR + shape)

| Model | AUROC mean ± std | AUPRC | Bal. acc | ΔAUROC vs age | Beats age? |
| --- | ---: | ---: | ---: | ---: | --- |
| Age + sex LR | **0.905 ± 0.025** | 0.747 | 0.851 | 0 | — |
| Radiomics LR | **0.954 ± 0.017** | 0.874 | 0.883 | **+0.049** | yes |
| Radiomics + clinical LR | **0.969 ± 0.013** | 0.919 | 0.908 | **+0.064** | yes |
| Radiomics + clinical XGB | **0.965 ± 0.009** | 0.922 | 0.879 | **+0.060** | yes |

**Selected development baseline (for later external freeze):** radiomics + clinical LR (highest mean AUROC).

BraTS21 overlap among UCSF labels: **298/501** (`data/splits/ucsf_brats21_overlap.json`).

## Exploratory UTSW 2.5D (not confirmatory)

| Metric | Value |
| --- | ---: |
| 5-fold mean AUROC | 0.854 ± 0.030 |
| Age+sex LR (UTSW) | 0.875 ± 0.024 |

## In progress on Kaggle

- Tumor-centered crops: https://www.kaggle.com/code/itxsamad/btlab-paper-b-crops  
- Exploratory UTSW radiomics: https://www.kaggle.com/code/itxsamad/btlab-paper-b-radiomics-utsw  

## Next (protocol order)

1. Finish UCSF crops → train crop 3D CNN ± clinical (must appear in table)  
2. Optional: richer radiomics / full 4-modality on larger UCSF mirror  
3. Git tag `paper-b-external-freeze` with frozen UCSF model  
4. **Single** confirmatory UTSW external evaluation (no further tuning)  
