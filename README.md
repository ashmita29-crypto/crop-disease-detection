# Crop Disease Detection using CNN and Transfer Learning

## Project Overview
An AI system that detects plant diseases from leaf images using
Deep Learning. Built on the PlantVillage dataset with 15 disease
and health categories across multiple plant species.

## Team
- Member 1 (Ashmita)  : Model Integration, Training Pipeline, Evaluation
- Member 2 (Advait)   : Transfer Learning, Hyperparameter Tuning
- Member 3 (Deva)     : Inference System, Documentation, Repository Cleanup

## Tech Stack
Python · TensorFlow · Keras · MobileNetV2 · NumPy · Pillow
Matplotlib · Seaborn · Scikit-learn · OpenCV

## Results Summary

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Custom CNN (Week 2) | 97.19% | 0.0838 |
| MobileNetV2 (Week 3) | 96.96% | 0.0914 |

## Week 1 — Completed ✅
- [x] GitHub repository setup with 3 contributors
- [x] PlantVillage dataset downloaded and organised (15 classes)
- [x] Exploratory Data Analysis — class distribution, sample images
- [x] Image preprocessing — resized to 224x224, normalised to [0,1]
- [x] Train / Validation / Test split — 70% / 15% / 15%

## Week 2 — Completed ✅
- [x] Custom CNN architecture — Conv2D, MaxPooling, Dense, Dropout
- [x] Model compiled with Adam optimizer
- [x] EarlyStopping and ModelCheckpoint callbacks
- [x] Model trained — 97.19% test accuracy
- [x] Training accuracy and loss curves saved

## Week 3 — Completed ✅
- [x] MobileNetV2 transfer learning implemented
- [x] Phase 1 — frozen base, custom head trained
- [x] Phase 2 — fine-tuning top layers
- [x] Hyperparameter tuning — learning rate, batch size, dropout
- [x] CNN vs MobileNetV2 comparison completed

## Week 4 — Completed ✅
- [x] Confusion matrix generated
- [x] Per-class classification report (precision, recall, F1)
- [x] Live inference script — predict.py
- [x] Model comparison table
- [x] Final documentation and repository cleanup

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the custom CNN
```bash
python src/train.py
```

### Train MobileNetV2 transfer learning model
```bash
python src/train_transfer.py
```

### Predict disease from a leaf image
```bash
python src/predict.py images/test_leaf.jpg
```

### Generate evaluation reports
```bash
python src/confusion_matrix.py
python src/classification_report_gen.py
python src/compare_models.py
```

## Inference Example

Input  : python src/predict.py images/test_leaf.jpg
Output :
CROP DISEASE DETECTION RESULT
Predicted Disease : Tomato___Early_blight
Confidence        : 98.73%
Top 3 Predictions :
1. Tomato___Early_blight     98.73%
2. Tomato___Late_blight       0.81%
3. Tomato___Leaf_Mold         0.46%


## Project Structure

crop-disease-detection/
├── data/
│   └── splits/
│       ├── train/
│       ├── val/
│       └── test/
├── images/          ← place test images here
├── models/          ← saved trained models
├── reports/         ← EDA charts and outputs
├── results/         ← evaluation outputs
├── src/             ← all Python scripts
├── requirements.txt
└── README.md

