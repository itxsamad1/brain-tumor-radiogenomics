# Paper B — Progress Report

**Status:** Analysis plan **locked** for confirmatory claims; UCSF development radiomics + crops running on Kaggle  
**Question:** Calibrated, overlap-safe IDH / 1p19q / grade prediction across cohorts

Plan: [`docs/analysis_plan.md`](../../docs/analysis_plan.md)

## Scientific protocol (correct roles)

| Role | Cohort | Status |
| --- | --- | --- |
| Development CV | **UCSF-PDGM** | Labels + BraTS overlap registry ready; radiomics/crops kernels launched |
| Primary external (once) | **UTSW-Glioma** | Labels ready; **not** used for confirmatory eval until UCSF model freeze |
| Exploratory only | UTSW 2.5D PNG ResNet | Done — mean AUROC **0.854** vs age **0.875** (does **not** beat age) |

Do **not** rebrand exploratory UTSW 2.5D numbers as the locked external result.

## Clinical must-beat baselines (age + sex LR)

| Cohort | n IDH | AUROC mean ± std | Bal. acc |
| --- | ---: | ---: | ---: |
| UTSW | 622 | **0.875 ± 0.024** | 0.809 |
| UCSF | 501 (103 mut / 398 wt) | **0.904 ± 0.025** | 0.850 |

UCSF BraTS21 overlap: **298 / 501** (`data/splits/ucsf_brats21_overlap.json`).

## Exploratory imaging (UTSW 2.5D) — pipeline validation only

| Fold | Best AUROC |
| --- | ---: |
| 1–5 mean | **0.854 ± 0.030** |

## Confirmatory stack (in progress)

1. **Radiomics must-beat table** on UCSF (age LR vs radiomics LR vs radiomics+clinical LR/XGB) — Kaggle: `btlab-paper-b-radiomics`  
2. **Tumor-centered float16 crops** on UCSF — Kaggle: `btlab-paper-b-crops`  
3. Crop CNN ± clinical fusion on UCSF CV  
4. Git tag `paper-b-external-freeze` → **single** UTSW external run  

## Code map

| Path | Purpose |
| --- | --- |
| `src/btlab/radiomics/features.py` | Shape + first-order radiomics; tumor crops |
| `src/btlab/data/nifti_cohorts.py` | UTSW/UCSF NIfTI path resolvers |
| `src/btlab/data/overlap.py` | BraTS overlap registry helper |
| `scripts/paper_b_radiomics_baseline.py` | Honest CV must-beat table |
| `scripts/paper_b_make_crops.py` | Crop export |
| `docs/analysis_plan.md` | Locked confirmatory protocol |
