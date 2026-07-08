# Model Comparison Table

| Metric | Custom CNN (Week 2) | MobileNetV2 (Week 3) |
|---|---|---|
| Test Accuracy | 97.19% | 96.96% |
| Test Loss | 0.0838 | 0.0914 |
| Total Parameters | ~10M | ~3.4M (head only) |
| Training Epochs | 30 (max) | 20 (10+10 phases) |
| Data Augmentation | Yes | Yes |
| Preprocessing | Rescale /255 | MobileNetV2 preprocess |
| Overfitting | Low | Low |
| Architecture | Built from scratch | Transfer Learning |

**Conclusion:**
Both models perform nearly identically on our 15-class dataset.
The Custom CNN achieved marginally higher accuracy (97.19% vs 96.96%).
However MobileNetV2 converged faster using transfer learning,
demonstrating its efficiency advantage on smaller datasets.