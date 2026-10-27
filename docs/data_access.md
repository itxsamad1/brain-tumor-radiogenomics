# Data access (official sources)

Raw MRI is downloaded **outside** git. Always cite the collection DOI in papers.

## Priority order (download)

1. **Figshare — Jun Cheng** (~839 MB) — Paper A  
   https://doi.org/10.6084/m9.figshare.1512427  
   License: CC BY 4.0

2. **UTSW-Glioma** (~22.9 GB) — primary external test (Papers B/D)  
   https://www.cancerimagingarchive.net/collection/utsw-glioma/  
   DOI: https://doi.org/10.7937/dfae-1b86

3. **UCSF-PDGM** (~142 GB NIfTI) — development cohort  
   https://www.cancerimagingarchive.net/collection/ucsf-pdgm/  
   DOI: https://doi.org/10.7937/tcia.bdgf-8v37  
   Note: use official BraTS ID mapping to avoid leakage with BraTS 2021.

4. **UPenn-GBM** (NIfTI ~69 GB) — survival + prevalence stress test  
   https://www.cancerimagingarchive.net/collection/upenn-gbm/  
   DOI: https://doi.org/10.7937/TCIA.709X-DN49

5. **BraTS 2021** (segmentation NIfTI + TCIA ID crosswalk)  
   https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/  
   DOI: https://doi.org/10.7937/jc8x-9874

6. **TCGA-GBM / TCGA-LGG** + BraTS-TCGA segmentations + molecular labels  
   Use BraTS-TCGA labels; molecular from Ceccarelli / cBioPortal. Check BraTS overlap.

## Optional

- **EGD** (Erasmus Glioma Database) — DUA required: https://xnat.bmia.nl/  
- **LGG-1p19qDeletion** — TCIA  
- **RHUH-GBM** — TCIA  

## TCIA download notes

- Many NIfTI packages need the **IBM Aspera Connect** browser plugin.
- DICOM collections use the **NBIA Data Retriever**.
- Prefer downloading to `D:\Brain-Tumor\data\raw\<collection>/`.

## Kaggle packaging

Do **not** upload full raw UCSF/UPenn dumps (size + license care).

Upload only:

- float16 tumor-centered crops + masks  
- patient-level CSV labels / splits  
- overlap exclusion lists  

See `scripts/package_kaggle_dataset.py`.
