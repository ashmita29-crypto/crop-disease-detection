"""
compare_models.py

Loads both trained models (Week 2 CNN and Week 3 MobileNetV2)
and compares their performance on the test set.

Generates:
  results/model_comparison.png  — side-by-side accuracy bar chart
  results/comparison_report.txt — text summary of both models
"""

import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

SPLITS_DIR   = "data/splits"
RESULTS_DIR  = "results"
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32

os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_model(model_path, preprocessing='normalize'):
    """
    Loads a model and evaluates it on the test set.
    
    Args:
        model_path     : path to the .h5 model file
        preprocessing  : 'normalize' for CNN, 'mobilenet' for MobileNetV2
    
    Returns:
        (test_loss, test_accuracy)
    """
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, None
    
    model = tf.keras.models.load_model(model_path)
    
    if preprocessing == 'mobilenet':
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0/255.0
        )
    
    test_gen = datagen.flow_from_directory(
        os.path.join(SPLITS_DIR, "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    return loss, accuracy


print("Evaluating Week 2 Custom CNN...")
cnn_loss, cnn_acc = evaluate_model(
    "models/crop_disease_cnn.h5",
    preprocessing='normalize'
)

print("\nEvaluating Week 3 MobileNetV2...")
mob_loss, mob_acc = evaluate_model(
    "models/mobilenet_crop_disease.h5",
    preprocessing='mobilenet'
)

# ============================================================
# GENERATE COMPARISON CHART
# ============================================================
models     = ['Custom CNN\n(Week 2)', 'MobileNetV2\n(Week 3)']
accuracies = [cnn_acc * 100 if cnn_acc else 0,
              mob_acc * 100 if mob_acc else 0]
losses     = [cnn_loss if cnn_loss else 0,
              mob_loss if mob_loss else 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy bar chart
bars = ax1.bar(models, accuracies, color=['steelblue', 'darkorange'],
               width=0.4, edgecolor='white')
ax1.set_title("Test Accuracy Comparison", fontsize=14)
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(0, 105)
for bar, val in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.2f}%', ha='center', fontsize=12, fontweight='bold')

# Loss bar chart
bars2 = ax2.bar(models, losses, color=['steelblue', 'darkorange'],
                width=0.4, edgecolor='white')
ax2.set_title("Test Loss Comparison", fontsize=14)
ax2.set_ylabel("Loss")
for bar, val in zip(bars2, losses):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, "model_comparison.png")
plt.savefig(save_path, dpi=150)
plt.close()
print(f"\nComparison chart saved: {save_path}")

# ============================================================
# SAVE TEXT REPORT
# ============================================================
report = f"""MODEL COMPARISON REPORT
{'='*45}
Custom CNN (Week 2)
  Test Accuracy : {f'{cnn_acc*100:.2f}%' if cnn_acc else 'N/A'}
  Test Loss     : {f'{cnn_loss:.4f}' if cnn_loss else 'N/A'}

MobileNetV2 (Week 3)
  Test Accuracy : {f'{mob_acc*100:.2f}%' if mob_acc else 'N/A'}
  Test Loss     : {f'{mob_loss:.4f}' if mob_loss else 'N/A'}

Winner        : {'MobileNetV2' if mob_acc and cnn_acc and mob_acc > cnn_acc else 'Custom CNN'}
Improvement   : {abs((mob_acc or 0) - (cnn_acc or 0))*100:.2f}%
{'='*45}
"""