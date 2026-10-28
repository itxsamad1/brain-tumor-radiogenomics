# Paper B — Progress Report

**Status:** Started (after Paper A push)  
**Question:** Calibrated, overlap-safe IDH / 1p19q / grade prediction across cohorts

## Done so far

| Item | Status |
| --- | --- |
| Official UTSW clinical TSV (625 pts) | `data/raw/utsw-glioma/UTSW_Glioma_Metadata.tsv` |
| Frozen patient folds (IDH-stratified) | `data/splits/utsw_paper_b_folds.csv` |
| Age±sex logistic baseline | `experiments/paper_b/age_baseline/` |
| Kaggle 2.5D IDH kernel | `kaggle/kernels/btlab-paper-b-idh` |

## Label snapshot (UTSW)

- IDH: ~176 mutant / ~446 wild-type / 3 missing  
- 1p/19q: sparse outside IDH-mutant (task gated)  
- Primary external role in final design; early GPU work uses UTSW for pipeline proof while UCSF-PDGM downloads

## Next

1. Kaggle GPU fold runs (2.5D ResNet)  
2. Official Aspera NIfTI + UCSF-PDGM packaging  
3. Radiomics / TabPFN baselines; BraTS overlap map  
4. Lock `docs/analysis_plan.md` before external peeking  
