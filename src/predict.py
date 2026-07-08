"""
predict.py

Inference script for crop disease detection.

Usage:
    python src/predict.py path/to/your/image.jpg

Example:
    python src/predict.py images/test_leaf.jpg

Output:
    Predicted Disease : Tomato___Early_blight
    Confidence        : 98.73%
    Top 3 Predictions :
      1. Tomato___Early_blight       98.73%
      2. Tomato___Late_blight         0.81%
      3. Tomato___Leaf_Mold           0.46%
"""

import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# SETTINGS — change MODEL_PATH if needed
# ============================================================
MODEL_PATH   = "models/mobilenet_crop_disease.h5"
IMAGE_SIZE   = (224, 224)
SPLITS_DIR   = "data/splits/train"   # used to read class names

# ============================================================
# LOAD CLASS NAMES
# ============================================================
def get_class_names(train_dir):
    """
    Reads class names from the training folder structure.
    Each subfolder name = one disease class.
    """
    if not os.path.exists(train_dir):
        # Fallback: return numbered classes
        return [f"Class_{i}" for i in range(38)]
    classes = sorted(os.listdir(train_dir))
    return [c for c in classes if os.path.isdir(
        os.path.join(train_dir, c)
    )]

# ============================================================
# PREPROCESS IMAGE
# ============================================================
def preprocess_image(image_path):
    """
    Loads and preprocesses a single image for prediction.
    
    Steps:
    1. Open the image from disk
    2. Convert to RGB (handles grayscale or RGBA images)
    3. Resize to 224x224 (what MobileNetV2 expects)
    4. Convert to numpy array
    5. Apply MobileNetV2's preprocessing (scales to [-1, 1])
    6. Add batch dimension: (224,224,3) -> (1,224,224,3)
    """
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at '{image_path}'")
        sys.exit(1)
    
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    return img_array

# ============================================================
# PREDICT
# ============================================================
def predict(image_path):
    """
    Full prediction pipeline for a single image.
    
    Args:
        image_path: path to the image file
    
    Returns:
        predicted_class (str), confidence (float), all_probs (array)
    """
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at '{MODEL_PATH}'")
        print("Please run train_transfer.py first to train the model.")
        sys.exit(1)
    
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load class names
    class_names = get_class_names(SPLITS_DIR)
    
    # Preprocess image
    print(f"Processing image: {image_path}")
    img_array = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]  # Remove batch dimension
    
    # Get top prediction
    top_idx    = np.argmax(probabilities)
    confidence = probabilities[top_idx] * 100
    predicted_class = class_names[top_idx]
    
    # Get top 3 predictions
    top3_idx = np.argsort(probabilities)[::-1][:3]
    
    # ============================================================
    # DISPLAY RESULTS
    # ============================================================
    print("\n" + "="*50)
    print("  CROP DISEASE DETECTION RESULT")
    print("="*50)
    print(f"  Image             : {os.path.basename(image_path)}")
    print(f"  Predicted Disease : {predicted_class}")
    print(f"  Confidence        : {confidence:.2f}%")
    print("-"*50)
    print("  Top 3 Predictions :")
    for rank, idx in enumerate(top3_idx, 1):
        name = class_names[idx]
        prob = probabilities[idx] * 100
        print(f"    {rank}. {name:<45} {prob:.2f}%")
    print("="*50)
    
    return predicted_class, confidence, probabilities


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        print("Example: python src/predict.py images/test_leaf.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predicted_class, confidence, _ = predict(image_path)