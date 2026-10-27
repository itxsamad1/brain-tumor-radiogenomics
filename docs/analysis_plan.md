# Pre-registered analysis plan (draft)

Status: **draft** — lock before first external-test evaluation.

## Primary research question (Paper B)

Can a multi-task MRI model predict glioma molecular status (IDH; 1p/19q within IDH-mutant; WHO CNS5 type/grade) with calibrated probabilities and generalize across independent cohorts after enforcing patient-level splits and BraTS–TCIA overlap exclusion?

## Development / external cohorts

| Role | Cohort |
| --- | --- |
| Development (5-fold patient CV) | UCSF-PDGM |
| Primary external | UTSW-Glioma |
| Secondary external | TCGA (BraTS-TCGA labels) |
| Prevalence stress test | UPenn-GBM |
| Optional | EGD, UCSD-PTGBM |

## Overlap rule

Any patient appearing in BraTS 2021 training (via official mapping) is excluded from external-test claims for models that used BraTS pretraining or from UCSF subsets overlapping BraTS when those cases were used for training.

## Metrics (locked)

- Discrimination: AUROC, AUPRC, balanced accuracy  
- Operating point: threshold chosen on development validation fold only  
- Calibration: ECE, Brier  
- Uncertainty: risk–coverage; conformal “refer to biopsy” rate  
- Statistics: patient-level bootstrap 2000×; DeLong for paired AUROC; Holm correction  

## Baselines that must be beaten

Age-only LR; radiomics + LR / XGBoost / TabPFN; frozen foundation-model probes; 3D CNN; multi-task SwinUNETR-style model.

## External evaluation

Run **once** after analysis plan is committed to git (timestamped). No iterative peeking on external labels.
