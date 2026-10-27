# Paper A — Progress Report

**Status:** Phase 1 in progress (smoke train + fold-1 full eval + duplicate audit done; 5-fold + leaky split + Grad-CAM remaining).

**Plan:** [`docs/PAPER_PUBLISH_PLAN.md`](../../docs/PAPER_PUBLISH_PLAN.md)

## Completed

| Item | Result |
| --- | --- |
| Patient-level frozen splits | `data/splits/figshare_cvind_patient.csv` (233 patients, 3064 slices) |
| Kaggle dataset | `itxsamad/btlab-figshare-paper-a` |
| Smoke train (fold 1, ResNet-18, 30 epochs) | Best macro-F1 ≈ **0.962** |
| Full fold-1 eval + figures | `experiments/paper_a/fold1_resnet18/` |
| dHash near-duplicate audit | 5 cross-patient identical-hash groups — see `audits/` |

## Figures (fold 1)

- `fold1_resnet18/figures/confusion_matrix.png`
- `fold1_resnet18/figures/roc.png`
- `fold1_resnet18/figures/pr.png`
- `fold1_resnet18/figures/reliability.png`
- `fold1_resnet18/figures/training_curves.png`

## Remaining for publish-ready Paper A

1. Train folds 2–5 (and ResNet-50) on Kaggle — kernel `btlab-paper-a-full`.
2. Slice-level leaky baseline (same model) for the main comparison figure.
3. Tumor-masked control + Grad-CAM panels.
4. Aggregate 5-fold mean ± std table + bootstrap CIs.
5. Draft manuscript Methods/Results.

## How to reproduce fold-1 eval locally

```bash
uv run python scripts/paper_a_evaluate.py \
  --checkpoint data/kaggle_upload/paper_a_out/checkpoints/checkpoint_best.pt \
  --data-root data/kaggle_upload/btlab-figshare-paper-a \
  --splits data/kaggle_upload/btlab-figshare-paper-a/splits.csv \
  --out-dir experiments/paper_a/fold1_resnet18 \
  --fold 1 --use-png --cpu
```
