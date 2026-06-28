"""
preprocess.py

This is the MAIN preprocessing file that combines:
  1. Loading an image
  2. Resizing it to 224x224
  3. Normalising pixel values to [0, 1]

Other scripts will import the `preprocess_image()` function from this file.
"""

import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Full preprocessing pipeline for a single image.
    
    Args:
        image_path: path to the image file
        target_size: tuple (width, height) — default is (224, 224)
    
    Returns:
        A numpy array of shape (224, 224, 3) with values in [0, 1]
    """
    # Step 1: Open the image
    img = Image.open(image_path)
    
    # Step 2: Convert to RGB (some images might be grayscale or have 4 channels)
    img = img.convert("RGB")
    
    # Step 3: Resize to 224x224
    img = img.resize(target_size)
    
    # Step 4: Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Step 5: Normalise pixel values from [0, 255] to [0.0, 1.0]
    img_array = img_array / 255.0
    
    return img_array


# --- Test the pipeline ---
if __name__ == "__main__":
    import os
    
    processed_dir = "data/processed"
    
    # Find any image to test with
    for class_name in os.listdir(processed_dir):
        class_path = os.path.join(processed_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if images:
                test_path = os.path.join(class_path, images[0])
                result = preprocess_image(test_path)
                print("preprocess_image() test result:")
                print(f"  Input file  : {test_path}")
                print(f"  Output shape: {result.shape}")
                print(f"  Min value   : {result.min():.4f}")
                print(f"  Max value   : {result.max():.4f}")
                break