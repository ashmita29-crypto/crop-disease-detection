"""
evaluate_model.py

Loads the saved model and evaluates it on the test set.
The test set was never seen during training — these results
show real-world performance.
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH  = "models/crop_disease_cnn.h5"
TEST_DIR    = "data/splits/test"
REPORTS_DIR = "reports"
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32

def evaluate():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    print(f"\nTest Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save results
    with open(os.path.join(REPORTS_DIR, "test_results.txt"), "w") as f:
        f.write(f"Test Loss    : {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    
    print(f"\nResults saved to reports/test_results.txt")

if __name__ == "__main__":
    evaluate()