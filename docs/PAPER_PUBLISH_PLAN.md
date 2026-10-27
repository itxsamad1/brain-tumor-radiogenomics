# Paper & Publish Plan

**Project:** [brain-tumor-radiogenomics](https://github.com/itxsamad1/brain-tumor-radiogenomics)  
**Goal:** Publish high-quality, leakage-free glioma MRI research — not another saturated “tumor vs no-tumor CNN” paper.  
**Last updated:** 2026-10-27  

This document is the single source of truth for what we will build, in what order, with which datasets, metrics, figures, and publication targets.

---

## 1. Guiding principles

1. **MRI is the modality.** Brain tumors are researched with MRI (sometimes CT/PET), not plain X-ray.
2. **There is no single “internationally approved” dataset.** Credibility comes from: public cohorts with DOIs, patient-level splits, reliable labels, overlap audits, and independent external testing.
3. **Do not chase 99% accuracy on clean Kaggle 2D sets.** Reviewers reject that. Prefer hard clinical problems: radiogenomics, robustness, calibration, explainability.
4. **Patient-level splits only** for primary claims. Never put slices from the same patient in both train and test.
5. **External-test patients never appear in any training stage** (including BraTS pretraining), after checking official BraTS↔TCIA mappings.
6. **Report more than accuracy:** AUROC, AUPRC, macro-F1, per-class recall, confusion matrices, ECE/calibration, bootstrap CIs; for segmentation Dice + HD95; for survival C-index + calibration.
7. **Compute policy:** Kaggle free GPU (~30 h/week) for training while the local GPU runs other work; local CPU for prep, radiomics, audits, figure generation.
8. **Finish one paper package completely** (experiments + figures + tables + draft) before claiming it is publish-ready.

---

## 2. Publication portfolio (ordered)

| ID | Working title (angle) | Role | Compute | Est. when ready |
| --- | --- | --- | --- | --- |
| **A** | Reality check: leakage and shortcuts in 2D brain-tumor classification | Quick methodology / meta-science paper | Low (Kaggle + CPU) | Phase 1 — **do this first** |
| **B** | Trustworthy cross-cohort glioma radiogenomics (IDH / 1p19q / WHO CNS5) | **Flagship** clinical AI paper | Medium–high | Phase 2 |
| **C** | Imaging + clinical survival models for IDH-wildtype GBM | Survival / multimodal | Mostly CPU | Phase 2–3 |
| **D** | Segmentation robustness across scanners, field strength, missing modalities | Technical / clinical translation | High | Phase 3 |
| **E** (optional) | Foundation-model adaptation / label efficiency | If GPU time remains | Medium | Optional |

We do **not** submit a plain “ResNet on Figshare gets 96%” paper as the main contribution.

---

## 3. Datasets (inventory, roles, quality)

### 3.1 Currently available (downloaded / on Kaggle)

| Dataset | Scale | Labels | Quality notes | Role |
| --- | --- | --- | --- | --- |
| **Figshare — Jun Cheng** | 3,064 T1c 2D slices; **233 patients**; 3 classes (meningioma / glioma / pituitary) | Slice labels + `cvind.mat` folds; tumor masks | Clean, small, **2D only**; heavily used in weak papers | **Paper A only** |
| Kaggle pack `itxsamad/btlab-figshare-paper-a` | Same data as PNG (~286 MB) + frozen patient folds | Same | Convenient for GPU runs | Paper A training |

### 3.2 Required next (not yet fully downloaded)

| Dataset | Approx. size | Subjects | Why it matters | Papers |
| --- | --- | --- | --- | --- |
| **UTSW-Glioma** (TCIA, 2026) | ~23 GB | 625 | Multi-vendor / 0.3–3 T; IDH, 1p19q, MGMT; segmentations | B, D external |
| **UCSF-PDGM** (TCIA) | ~142 GB NIfTI | ~495 | Pre-op mpMRI + molecular + survival; BraTS overlap map exists | B, C development |
| **UPenn-GBM** (TCIA) | ~69 GB NIfTI | 630 | GBM outcomes, genomics, radiomics; extreme IDH imbalance | B stress test, C |
| **BraTS 2021** (+ ID crosswalk) | ~12 GB+ seg NIfTI | ~1,251 labeled | Segmentation benchmark; **overlaps** UCSF/UPenn/TCGA | D train; B encoder pretrain (overlap-safe) |
| **TCGA-GBM/LGG** + BraTS-TCGA labels | Smaller | ~240 | Molecular + survival; check BraTS overlap | B, C external |

### 3.3 Optional / DUA

- **EGD** (Erasmus) — strong balanced external for IDH; requires data-use agreement.  
- **LGG-1p19qDeletion**, **RHUH-GBM**, **UCSD-PTGBM** (post-treatment shift).

### 3.4 Hard overlap rule

BraTS 2021 includes patients from UCSF-PDGM, UPenn-GBM, TCGA, IvyGAP, CPTAC, etc.  
Before any “external validation” claim:

1. Apply official BraTS↔TCIA / UCSF BraTS-ID mappings.  
2. Content-based duplicate checks where needed.  
3. Freeze a patient registry: each patient has exactly one role per study.

---

## 4. End-to-end research flow

```text
Phase 0  Repo + docs + Kaggle auth + Figshare download
    │
Phase 1  PAPER A (complete package)  ← we are here / finishing this
    │      patient splits, multi-model, audits, full figures, draft
    │
Phase 2  Download TCIA (UTSW → UCSF → UPenn → BraTS)
    │      preprocess crops → Kaggle datasets
    │
Phase 3  PAPER B flagship radiogenomics
    │      baselines + multi-task model + external tests + XAI + calibration
    │
Phase 4  PAPER C survival  |  PAPER D segmentation robustness
    │
Phase 5  Writing, checklists (CLAIM 2024, TRIPOD+AI), code/weights release
```

### Compute flow

```text
Local CPU (low priority)          Kaggle GPU (~30 h/week)
─────────────────────────         ────────────────────────
Downloads, unzip, PNG/crops       Train / resume epochs
Radiomics, survival baselines     5-fold / multi-seed jobs
Audits, statistics, plots         Heavy 3D / nnU-Net later
Sync checkpoints ←→               Write checkpoint_last.pt each epoch
```

**Resume rule:** every epoch saves `checkpoint_last.pt` / `checkpoint_best.pt`. If a 12 h session ends, sync outputs into `itxsamad/btlab-paper-a-ckpts` (or Paper B ckpt dataset) and re-run; training continues from the last epoch.

---

## 5. Paper A — full specification (Phase 1)

### 5.1 Research question

How much of the widely reported 98–99% accuracy on Figshare-style 2D brain-tumor classification comes from **data leakage** (slice-level splits, near-duplicates) and **shortcuts** (imaging plane confounding), versus genuine patient-level generalization?

### 5.2 Data & splits

- Source: Figshare Cheng (CC BY 4.0); cite DOI `10.6084/m9.figshare.1512427`.  
- Frozen file: `data/splits/figshare_cvind_patient.csv` (patient-majority fold from official `cvind`).  
- Primary protocol: **patient-level** 5-fold CV.  
- Comparison protocol: **slice-level random** split (intentionally leaky) with the same models.

### 5.3 Models

- ResNet-18, ResNet-50 (ImageNet init).  
- Optional: EfficientNet-B0 / ConvNeXt-Tiny if time allows.  
- Seed(s): 42 (primary); 0 and 1 for sensitivity on main model.

### 5.4 Metrics (must report)

- Accuracy, **macro-F1**, balanced accuracy  
- Per-class precision / recall / F1  
- Confusion matrix  
- One-vs-rest ROC + AUROC; PR + AUPRC  
- Expected Calibration Error (ECE), reliability diagram  
- Patient-level bootstrap 95% CIs on macro-F1 / AUROC  

### 5.5 Audits (must report)

1. **Split audit:** patient-level vs slice-level performance gap (main figure).  
2. **Near-duplicate audit:** perceptual hash / embedding similarity across folds.  
3. **Imaging-plane shortcut:** class vs plane confounding; plane-only baseline if plane metadata available; otherwise orientation proxy from image shape / acquisition notes in README.  
4. **Tumor-masked control:** train/eval with tumor region zeroed — drop in performance shows reliance on tumor vs background.  
5. **Literature mini-table:** recent papers’ split protocols (patient vs slice).

### 5.6 Figures & tables for the manuscript

| ID | Content |
| --- | --- |
| Fig 1 | Study flowchart (data → splits → models → audits) |
| Fig 2 | Class / patient counts; fold balance |
| Fig 3 | Patient-level vs slice-level metrics (bar + CI) |
| Fig 4 | Confusion matrices (best patient-level model) |
| Fig 5 | ROC / PR curves |
| Fig 6 | Calibration / reliability diagram |
| Fig 7 | Training curves (loss, macro-F1) |
| Fig 8 | Grad-CAM / attention examples (correct + failure) |
| Fig 9 | Audit summary (duplicates / masked / shortcuts) |
| Table 1 | Dataset summary |
| Table 2 | 5-fold results per model (mean ± std) |
| Table 3 | Ablation / audit results |
| Table 4 | Comparison to prior Figshare papers (split protocol called out) |

### 5.7 What we already ran (baseline smoke test)

- Kaggle kernel `itxsamad/btlab-paper-a-train`  
- ResNet-18, fold 1, 30 epochs  
- Best **macro-F1 ≈ 0.962**, acc ≈ 0.97 on patient-level fold-1 val  
- This is **not** the full Paper A package — only a pipeline proof.

### 5.8 Remaining Paper A work checklist

- [ ] Evaluation script: predictions → all metrics + all figures  
- [ ] 5-fold × ResNet-18 (and ResNet-50) on Kaggle with resume  
- [ ] Slice-level leaky baseline (same architecture)  
- [ ] Tumor-masked and near-duplicate audits  
- [ ] Grad-CAM figures  
- [ ] Aggregated `experiments/paper_a/REPORT.md` + `figures/`  
- [ ] Draft Methods / Results / Discussion for Paper A  

---

## 6. Paper B — flagship radiogenomics (after Paper A)

### 6.1 Question

Can a multi-task MRI model predict IDH (primary), 1p/19q within IDH-mutant, and WHO CNS5 type/grade with **calibrated** probabilities and **cross-cohort** generalization after overlap-safe patient splits?

### 6.2 Design

- **Dev:** UCSF-PDGM, 5-fold patient CV stratified by IDH + grade.  
- **External (locked, run once after pre-registration):** UTSW (primary), TCGA, UPenn (prevalence stress); optional EGD.  
- **Tasks:** segmentation (auxiliary) + hierarchical molecular / grade heads; modality dropout for missing sequences.  
- **Must-beat baselines:** age-only LR; radiomics + LR / XGBoost / TabPFN; frozen foundation probes; 3D CNN; multi-task SwinUNETR-style.  
- **Trust layer:** temperature scaling, label-shift adaptation, conformal “refer to biopsy”, risk–coverage.  
- **XAI:** compartment occlusion, Grad-CAM / IG with faithfulness checks.

### 6.3 Deliverables

Full metrics tables, cross-cohort heatmaps, calibration plots, XAI figures, CLAIM/TRIPOD+AI checklists, code + frozen splits + weights.

---

## 7. Paper C — GBM survival

- Cohorts: UPenn-GBM, UCSF IDH-wt GBM, TCGA-GBM (± RHUH).  
- Features: clinical ± radiomics ± Paper B embeddings; pre-op vs post-op (EOR) models.  
- Metrics: Harrell/Uno C-index, time-dependent AUC, IBS, calibration, KM risk groups.  
- Key claim: **does imaging add value beyond clinical covariates?** (ΔC-index + LRT).

---

## 8. Paper D — segmentation robustness

- Train: BraTS 2021 (overlap-safe).  
- Test: UTSW manual masks, UCSF non-BraTS holdout, RHUH.  
- nnU-Net baseline vs modality-dropout; uncertainty failure detection.  
- Metrics: lesion-wise Dice, HD95, failure-detection AUROC.

---

## 9. Rigor standard (all papers)

- Frozen, hashed patient splits before modeling.  
- Augmentation / normalization fit on train folds only.  
- External tests once after analysis plan committed (git timestamp ± OSF).  
- Statistics: patient bootstrap (2000×), DeLong for paired AUROC, Holm correction, ≥3 seeds for main models.  
- Label harmonization to WHO CNS5 documented (UCSF relative co-deletion, TCGA WHO-2016, obsolete oligoastrocytoma, MGMT assay differences).  
- Reporting: CLAIM 2024, TRIPOD+AI, METRICS where applicable.  
- Release: code, splits, overlap map, weights (dataset licenses respected).

---

## 10. Tooling map (this repo)

| Path | Purpose |
| --- | --- |
| `src/btlab/` | Library: data, train, eval, xai |
| `configs/` | Experiment YAML |
| `docs/PAPER_PUBLISH_PLAN.md` | This plan |
| `docs/analysis_plan.md` | Pre-registration draft (Paper B) |
| `docs/data_access.md` | Official download links / DOIs |
| `docs/gap_matrix.md` | Literature scoring stub |
| `kaggle/kernels/` | Pushable GPU scripts |
| `kaggle/datasets/` | Dataset metadata |
| `scripts/export_figshare_kaggle.py` | PNG packaging |
| `scripts/kaggle_push_and_run.py` | Create/push datasets & kernels |
| `scripts/kaggle_sync_checkpoints.py` | Resume sync after a run |
| `experiments/paper_a/` | Paper A outputs, figures, report |
| `data/splits/` | Frozen split CSVs (tracked) |
| `data/raw/` | Large downloads (gitignored) |

---

## 11. Kaggle resources (current)

- Dataset: https://www.kaggle.com/datasets/itxsamad/btlab-figshare-paper-a  
- Checkpoints: https://www.kaggle.com/datasets/itxsamad/btlab-paper-a-ckpts  
- Train kernel: https://www.kaggle.com/code/itxsamad/btlab-paper-a-train  
- GitHub: https://github.com/itxsamad1/brain-tumor-radiogenomics  

---

## 12. Target venues (indicative)

| Paper | Possible venues |
| --- | --- |
| A | Scientific Data / Patterns / MIDL short / workshop; or Methods section of a larger journal |
| B | Radiology: AI, Medical Image Analysis, Neuro-Oncology, MICCAI / MIDL |
| C | Neuro-Oncology / Cancers / Frontiers (survival focus) |
| D | MICCAI, IEEE TMI, Medical Physics |

Exact venue chosen after results exist — do not lock a journal before metrics and novelty are clear.

---

## 13. Immediate next actions (execution order)

1. **Write/maintain this plan** (done in this file).  
2. **Complete Paper A package:** full eval + figures + 5-fold + audits + `experiments/paper_a/REPORT.md`.  
3. Start TCIA downloads (UTSW first, then UCSF/UPenn/BraTS) for Papers B–D.  
4. Implement Paper B baselines on CPU while Kaggle runs 3D jobs.  
5. Lock analysis plan → single external evaluation → write flagship paper.

---

## 14. What “done” means before publishing Paper A

Paper A is **publish-ready** only when:

- All checklist items in §5.8 are checked.  
- Patient-level 5-fold results exist for ≥2 architectures.  
- Slice-level comparison and at least two audits are in the report.  
- Every figure/table in §5.6 is generated from scripts (reproducible).  
- Methods text can be written without hand-waving about splits.  
- Code and frozen splits are on GitHub; Figshare DOI cited.

Until then, high fold-1 F1 alone is only a **smoke test**, not a paper.
