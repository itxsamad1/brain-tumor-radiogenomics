# Paper A — Fold 4 evaluation

- Model: `resnet18`
- Checkpoint: `experiments\paper_a\fold4_resnet18\checkpoint_best.pt`
- Val slices: 628 | patients: 47

## Metrics
- Accuracy: **0.9745** (95% CI 0.9618–0.9841)
- Macro-F1: **0.9686** (95% CI 0.9520–0.9806)
- Balanced accuracy: **0.9719**
- AUROC (macro OVR): **0.9958**
- AUPRC (macro): **0.9886**
- ECE: **0.0210**

## Per-class
```
              precision    recall  f1-score   support

  meningioma     0.9154    0.9597    0.9370       124
      glioma     0.9876    0.9785    0.9830       325
   pituitary     0.9943    0.9777    0.9859       179

    accuracy                         0.9745       628
   macro avg     0.9658    0.9719    0.9686       628
weighted avg     0.9752    0.9745    0.9747       628

```

## Figures
- `figures/confusion_matrix.png`
- `figures/roc.png`, `figures/pr.png`
- `figures/reliability.png`
- `figures/training_curves.png`
