# Paper A — Fold 3 evaluation

- Model: `resnet18`
- Checkpoint: `experiments\paper_a\fold3_resnet18\checkpoint_best.pt`
- Val slices: 572 | patients: 48

## Metrics
- Accuracy: **0.9563** (95% CI 0.9406–0.9720)
- Macro-F1: **0.9522** (95% CI 0.9342–0.9699)
- Balanced accuracy: **0.9542**
- AUROC (macro OVR): **0.9923**
- AUPRC (macro): **0.9809**
- ECE: **0.0314**

## Per-class
```
              precision    recall  f1-score   support

  meningioma     0.9021    0.9281    0.9149       139
      glioma     0.9688    0.9394    0.9538       231
   pituitary     0.9805    0.9950    0.9877       202

    accuracy                         0.9563       572
   macro avg     0.9504    0.9542    0.9522       572
weighted avg     0.9567    0.9563    0.9563       572

```

## Figures
- `figures/confusion_matrix.png`
- `figures/roc.png`, `figures/pr.png`
- `figures/reliability.png`
- `figures/training_curves.png`
