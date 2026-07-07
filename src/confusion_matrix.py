"""
confusion_matrix.py

Generates a confusion matrix for the best trained model.

What is a confusion matrix?
- A grid where rows = actual disease, columns = predicted disease
- Diagonal = correct predictions (we want these HIGH)
- Off-diagonal = mistakes (we want these LOW)
- Tells us exactly which diseases the model confuses with each other
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SPLITS_DIR  = "data/splits"
RESULTS_DIR = "results"
MODEL_PATH  = "models/mobilenet_crop_disease.h5"
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load test data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)
test_gen = datagen.flow_from_directory(
    os.path.join(SPLITS_DIR, "test"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False      # IMPORTANT: must be False for confusion matrix
)

class_names = list(test_gen.class_indices.keys())
print(f"Classes: {len(class_names)}")

# Get predictions
print("Generating predictions...")
predictions = model.predict(test_gen, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes      = test_gen.classes

# Compute confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_classes, predicted_classes)

# ============================================================
# PLOT — Full confusion matrix (38x38)
# ============================================================
plt.figure(figsize=(24, 20))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.3
)
plt.title("Confusion Matrix — MobileNetV2 Crop Disease Detection",
          fontsize=16, pad=20)
plt.ylabel("True Label", fontsize=13)
plt.xlabel("Predicted Label", fontsize=13)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()

save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(save_path, dpi=150)
plt.close()
print(f"Confusion matrix saved: {save_path}")

# ============================================================
# OVERALL ACCURACY
# ============================================================
correct = np.trace(cm)
total   = np.sum(cm)
print(f"\nOverall accuracy: {correct/total*100:.2f}%")
print(f"Correct         : {correct} / {total}")