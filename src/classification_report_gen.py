"""
classification_report_gen.py

Generates a full classification report showing:
- Precision : of all predictions for a class, how many were correct?
- Recall    : of all actual instances of a class, how many did we find?
- F1-Score  : harmonic mean of precision and recall (overall quality)
- Support   : how many test images exist for each class

A good model should have all three above 0.90 for every class.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
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

print("Predicting...")
predictions     = model.predict(test_gen, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)
true_labels      = test_gen.classes

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    digits=4
)

print("\nCLASSIFICATION REPORT:")
print(report)

# Save to file
report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("CLASSIFICATION REPORT — MobileNetV2 Crop Disease Detection\n")
    f.write("="*65 + "\n\n")
    f.write(report)

print(f"Report saved: {report_path}")