# Paper A — Fold 5 evaluation

- Model: `resnet18`
- Checkpoint: `experiments\paper_a\fold5_resnet18\checkpoint_best.pt`
- Val slices: 643 | patients: 47

## Metrics
- Accuracy: **0.9393** (95% CI 0.9191–0.9565)
- Macro-F1: **0.9330** (95% CI 0.9106–0.9521)
- Balanced accuracy: **0.9362**
- AUROC (macro OVR): **0.9892**
- AUPRC (macro): **0.9762**
- ECE: **0.0428**

## Per-class
```
              precision    recall  f1-score   support

  meningioma     0.8855    0.8855    0.8855       166
      glioma     0.9855    0.9443    0.9644       287
   pituitary     0.9208    0.9789    0.9490       190

    accuracy                         0.9393       643
   macro avg     0.9306    0.9362    0.9330       643
weighted avg     0.9406    0.9393    0.9395       643

```

## Figures
- `figures/confusion_matrix.png`
- `figures/roc.png`, `figures/pr.png`
- `figures/reliability.png`
- `figures/training_curves.png`
