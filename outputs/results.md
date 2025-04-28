# Results Summary 📈

This project fine-tuned a pretrained ResNet-18 model on histopathology image patches (PatchCamelyon dataset) to classify tumor vs normal tissue.  
The pipeline was built to demonstrate end-to-end deep learning design, including data handling from .h5 files, model fine-tuning, structured training and evaluation, GradCAM interpretability, and production readiness for real-world biomedical AI tasks.

---

## Final Evaluation Metrics

- **Test Accuracy**: 0.8253
- **ROC-AUC**: 0.90

---

## Outputs Generated

- GradCAM heatmaps for test images: `outputs/gradcam/`
- ROC curve plot: `outputs/metrics/roc_curve.png`
- Confusion matrix plot: `outputs/metrics/confusion_matrix.png`
- Best model checkpoint: `models/best_model.pth`

---

## Key Takeaways

- Successful fine-tuning of ResNet-18 for biomedical image classification.
- Achieved strong classification performance with visual explainability (GradCAM).
- Fully modular, production-ready codebase supporting training, evaluation, and deployment extensions.

---
