"""
plot_training.py

Plots accuracy and loss curves after training.

How to read these charts:
- ACCURACY CURVES: Higher is better. Train accuracy should rise.
  Val accuracy should rise too. If train is much higher than val,
  the model is overfitting (memorising, not learning).
- LOSS CURVES: Lower is better. Both should fall over time.
  If val loss starts rising while train loss falls, that's overfitting.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_training_history(history, save_dir="reports"):
    """
    Takes the history object returned by model.fit() and creates two plots:
    1. Accuracy over epochs (train vs validation)
    2. Loss over epochs (train vs validation)
    
    Args:
        history: object returned by model.fit()
        save_dir: folder to save charts
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # --- PLOT 1: Accuracy ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.history['accuracy'],     label='Training Accuracy',   color='steelblue')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', color='darkorange')
    plt.title("Model Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_accuracy.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/training_accuracy.png")
    
    # --- PLOT 2: Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.history['loss'],     label='Training Loss',   color='steelblue')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss', color='darkorange')
    plt.title("Model Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_dir}/training_loss.png")