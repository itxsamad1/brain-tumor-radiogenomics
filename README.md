# brain-tumor-radiogenomics

Leakage-free, trustworthy MRI-based glioma radiogenomics research codebase.

**Focus:** noninvasive prediction of molecular markers (IDH, 1p/19q, WHO CNS5 type/grade) from multiparametric MRI, with patient-level splits, dataset-overlap audits, external validation, calibration, and explainability.

> This is **not** a tumor-vs-no-tumor Kaggle CNN project. We use clinical-grade public cohorts (BraTS, UCSF-PDGM, UPenn-GBM, UTSW-Glioma) and modern evaluation standards.

## Paper & publish plan

See **[`docs/PAPER_PUBLISH_PLAN.md`](docs/PAPER_PUBLISH_PLAN.md)** for the full English plan: paper portfolio (A–D), datasets, metrics, figures, Kaggle workflow, and what “publish-ready” means.

## Papers (planned)

| ID | Topic | Primary data |
| --- | --- | --- |
| A | 2D tumor-type classification leakage / shortcut audit | Figshare (Jun Cheng) |
| B | Cross-cohort radiogenomics (flagship) | UCSF-PDGM → UTSW / TCGA / UPenn |
| C | GBM overall survival (imaging + clinical) | UPenn-GBM, UCSF, TCGA |
| D | Segmentation robustness (scanner / field / missing modality) | BraTS → UTSW / UCSF holdout |

## Compute policy

- **Local GPU:** left free for other training when busy.
- **Training:** [Kaggle](https://www.kaggle.com/) free GPU (~30 h/week, P100 or 2× T4, 12 h sessions) with resumable checkpoints.
- **Local CPU:** preprocessing, radiomics, survival baselines at low priority.

## Quick start

```bash
# Local (Windows / Linux)
uv sync
uv run python -c "import btlab; print(btlab.__version__)"
```

Kaggle: clone this repo in a notebook, attach preprocessed crop datasets, run notebooks under [`notebooks/kaggle/`](notebooks/kaggle/).

## Data

Raw MRI is **not** in this repository. See [`docs/data_access.md`](docs/data_access.md) for official download links, licenses, and citation DOIs.

Place data under:

```text
data/raw/          # official downloads (gitignored)
data/registry/     # patient tables, overlap maps (tracked when small)
data/splits/       # frozen patient-level splits (tracked)
data/cache/        # float16 crops etc. (gitignored)
```

## Hard rules

1. **Patient-level** train/val/test splits only.
2. Augmentation / normalization fit on **train folds only**.
3. External-test patients never appear in any training stage (including BraTS pretraining).
4. Report AUROC, AUPRC, balanced accuracy, ECE — not accuracy alone.
5. Check BraTS ↔ TCIA overlap before claiming independent external tests.

## License

Code: MIT. Datasets retain their original licenses (typically CC BY 4.0 on TCIA CC-BY collections) — cite every collection you use.

## Citation

```bibtex
@software{brain_tumor_radiogenomics,
  title  = {brain-tumor-radiogenomics},
  author = {itxsamad1},
  year   = {2026},
  url    = {https://github.com/itxsamad1/brain-tumor-radiogenomics}
}
```
