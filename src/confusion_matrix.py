"""
confusion_matrix.py
Generates a confusion matrix for the best trained model.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SPLITS_DIR  = "data/splits"
RESULTS_DIR = "results"
MODEL_PATH  = "models/mobilenet_crop_disease.h5"
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)
test_gen = datagen.flow_from_directory(
    os.path.join(SPLITS_DIR, "test"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_gen.class_indices.keys())
print(f"Classes found: {len(class_names)}")

print("Generating predictions...")
predictions      = model.predict(test_gen, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes      = test_gen.classes

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(16, 14))
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
          fontsize=14, pad=20)
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(save_path, dpi=150)
plt.close()
print(f"Confusion matrix saved: {save_path}")

correct = np.trace(cm)
total   = np.sum(cm)
print(f"\nOverall accuracy : {correct/total*100:.2f}%")
print(f"Correct          : {correct} / {total}")
