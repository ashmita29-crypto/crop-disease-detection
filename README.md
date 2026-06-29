# Crop Disease Detection using CNN

## Project Overview
An AI system that detects plant diseases from leaf images using a
Convolutional Neural Network (CNN), built on the PlantVillage dataset
containing 54,000+ images across 38 disease/health categories.

## Team
- Member 1: Ashmita
- Member 2: Vradant
- Member 3: Advait

## Tech Stack
Python · TensorFlow · Keras · NumPy · Pillow · Matplotlib · Scikit-learn

## Week 1 — Completed ✅
- [x] GitHub repository setup and folder structure
- [x] PlantVillage dataset downloaded and organised
- [x] Exploratory Data Analysis (class distribution, sample images)
- [x] Image preprocessing — resized to 224×224, normalised to [0, 1]
- [x] Train / Validation / Test split — 70% / 15% / 15%

## Week 2 — Completed ✅
- [x] Custom CNN architecture built (Conv2D, MaxPooling, Dense, Dropout)
- [x] Model compiled with Adam optimizer and categorical crossentropy loss
- [x] EarlyStopping and ModelCheckpoint callbacks added
- [x] Model trained on PlantVillage training set
- [x] Training accuracy and loss curves saved to reports/
- [x] Model evaluated on test set
- [x] Trained model saved as models/crop_disease_cnn.h5

## Results
| Metric | Score |
|---|---|
| Test Accuracy | 97.19% |
| Test Loss | 0.0838 |

## Project Structure
crop-disease-detection/
├── data/
│   ├── raw/          # Original PlantVillage images
│   ├── processed/    # Resized 224x224 images
│   └── splits/       # Train / Val / Test split
├── src/              # All Python scripts
├── models/           # Saved trained model (.h5)
├── reports/          # Charts and result outputs
└── notebooks/        # Jupyter notebooks

## How to Run
# 1. Clone the repo
git clone https://github.com/ashmita29-crypto/crop-disease-detection.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python src/train.py

# 4. Evaluate on test set
python src/evaluate_model.py