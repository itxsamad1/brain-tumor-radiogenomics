# Paper A — Fold 2 evaluation

- Model: `resnet18`
- Checkpoint: `/kaggle/working/checkpoints/fold2_resnet18/checkpoint_best.pt`
- Val slices: 679 | patients: 46

## Metrics
- Accuracy: **0.9440** (95% CI 0.9264–0.9588)
- Macro-F1: **0.9401** (95% CI 0.9197–0.9564)
- Balanced accuracy: **0.9419**
- AUROC (macro OVR): **0.9934**
- AUPRC (macro): **0.9858**
- ECE: **0.0388**

## Per-class
```
              precision    recall  f1-score   support

  meningioma     0.9162    0.9162    0.9162       167
      glioma     0.9611    0.9497    0.9554       338
   pituitary     0.9382    0.9598    0.9489       174

    accuracy                         0.9440       679
   macro avg     0.9385    0.9419    0.9401       679
weighted avg     0.9442    0.9440    0.9441       679

```

## Figures
- `figures/confusion_matrix.png`
- `figures/roc.png`, `figures/pr.png`
- `figures/reliability.png`
- `figures/training_curves.png`
