# Paper A — Fold 1 evaluation

- Model: `resnet18`
- Checkpoint: `data\kaggle_upload\paper_a_out\checkpoints\checkpoint_best.pt`
- Val slices: 542 | patients: 45

## Metrics
- Accuracy: **0.9668** (95% CI 0.9520–0.9815)
- Macro-F1: **0.9615** (95% CI 0.9430–0.9775)
- Balanced accuracy: **0.9660**
- AUROC (macro OVR): **0.9984**
- AUPRC (macro): **0.9957**
- ECE: **0.0173**

## Per-class
```
              precision    recall  f1-score   support

  meningioma     0.9076    0.9643    0.9351       112
      glioma     0.9876    0.9714    0.9794       245
   pituitary     0.9780    0.9622    0.9700       185

    accuracy                         0.9668       542
   macro avg     0.9577    0.9660    0.9615       542
weighted avg     0.9678    0.9668    0.9671       542

```

## Figures
- `figures/confusion_matrix.png`
- `figures/roc.png`, `figures/pr.png`
- `figures/reliability.png`
- `figures/training_curves.png`
