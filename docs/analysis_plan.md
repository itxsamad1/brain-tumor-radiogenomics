# Pre-registered analysis plan — Paper B (radiogenomics)

**Status:** LOCKED for confirmatory claims as of commit timestamp on `main` (see git history for `docs/analysis_plan.md`).  
**Rule:** After this lock, primary external (UTSW) labels may be used for **one** confirmatory evaluation of the frozen UCSF-selected model(s). No iterative tuning on UTSW.

---

## 1. Primary research question

Can a multi-task MRI model predict glioma molecular status (**IDH** primary; **1p/19q within IDH-mutant**; WHO CNS5 type/grade) with **calibrated** probabilities and **cross-cohort** generalization after patient-level splits and BraTS–TCIA overlap exclusion?

### Primary estimand (confirmatory)

- Endpoint: IDH mutant vs wild-type discrimination (**AUROC**) on **UTSW-Glioma**, using a model selected only on **UCSF-PDGM** development CV.  
- Co-primary reporting: AUPRC, balanced accuracy, ECE/Brier, and **ΔAUROC vs age+sex logistic regression** (must show incremental value or honestly report absence).

---

## 2. Cohorts & roles

| Role | Cohort | DOI / source |
| --- | --- | --- |
| Development (5-fold patient CV) | UCSF-PDGM | 10.7937/tcia.bdgf-8v37 |
| Primary external (once) | UTSW-Glioma | 10.7937/DFAE-1B86 |
| Secondary external | TCGA (BraTS-TCGA labels) | TCIA / BraTS |
| Prevalence stress | UPenn-GBM | 10.7937/TCIA.709X-DN49 |

Frozen splits:

- `data/splits/ucsf_paper_b_folds.csv` (IDH-stratified, seed 42)  
- `data/splits/utsw_paper_b_folds.csv` (IDH-stratified, seed 42) — for **exploratory** UTSW-internal CV only until confirmatory freeze  
- Overlap registry: `data/splits/ucsf_brats21_overlap.json`

---

## 3. Overlap / leakage rules (hard)

1. Patient-level splits only (no slice-level train/test for primary claims).  
2. Any UCSF patient with a BraTS21 ID is flagged (`in_brats21`).  
3. Models that use BraTS 2021 pretraining **must not** count BraTS-overlapping UCSF cases as independent evidence, and must exclude mapped patients from external claims.  
4. Exploratory experiments on UTSW (e.g. early 2.5D PNG ResNet) are **pipeline validation only** — not confirmatory external results.

---

## 4. Targets

| Priority | Target | Notes |
| --- | --- | --- |
| Primary | IDH (binary) | Harmonize UCSF mutation strings → mutant; UTSW wild type/mutated |
| Secondary | 1p/19q | **Only within IDH-mutant**; report n explicitly |
| Secondary | WHO grade / type | CNS5-aligned labels where available |
| Exploratory | MGMT | High missingness — secondary only |

---

## 5. Models that must appear in the must-beat table

1. Age + sex logistic regression  
2. Radiomics + LR  
3. Radiomics + clinical LR  
4. Radiomics + clinical XGBoost (or TabPFN if license/compute allows)  
5. Tumor-centered 2.5D / 3D CNN (± clinical fusion)  
6. (Stretch) multi-task segmentation + molecular head  

Selection rule: choose the UCSF development model by **mean validation AUROC**, with tie-break on AUPRC then ECE. Then freeze weights/hyperparameters → single UTSW run.

---

## 6. Metrics (locked)

- Discrimination: AUROC, AUPRC, balanced accuracy  
- Operating point: threshold from development validation only (Youden or fixed sensitivity)  
- Calibration: ECE (15 bins), Brier  
- Incremental value: ΔAUROC vs age+sex; likelihood-ratio test on nested LR when applicable  
- Uncertainty (stretch): risk–coverage; conformal “refer to biopsy” rate  
- Statistics: patient bootstrap 2000× on external; DeLong for paired AUROC vs age; Holm for multi-endpoint  

---

## 7. What is already exploratory (do not rebrand as confirmatory)

- UTSW mid-slice 2.5D ResNet (~AUROC 0.85 mean) — **below** UTSW age LR (0.875)  
- Any UTSW tuning after seeing those numbers  

---

## 8. External evaluation procedure

1. Finish UCSF development must-beat table (radiomics + crops CNN).  
2. Freeze model card (git tag `paper-b-external-freeze`).  
3. Run **once** on UTSW; write results without further hyperparameter search.  
4. Optional secondary externals thereafter.

---

## 9. Reporting

CLAIM 2024 + TRIPOD+AI checklists; cite TCIA/Figshare DOIs; release code, frozen splits, overlap registry, and weights under dataset licenses.
