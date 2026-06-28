import numpy as np
from PIL import Image
import os

def load_and_normalise(image_path):
    """
    Load a single image and normalise pixel values to [0, 1].
    
    What normalisation means:
    - Original pixels: integers from 0 to 255
    - After dividing by 255: floats from 0.0 to 1.0
    - Neural networks train faster and more stably with small numbers
    """
    img = Image.open(image_path)
    img = img.convert("RGB")
    img_array = np.array(img, dtype=np.float32)  # Convert image to a matrix of numbers
    img_normalised = img_array / 255.0            # Divide every number by 255
    return img_normalised

def verify_normalisation():
    """Test that normalisation works correctly on one sample image."""
    # Find any image in processed folder
    processed_dir = "data/processed"
    
    for class_name in os.listdir(processed_dir):
        class_path = os.path.join(processed_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if images:
                sample_path = os.path.join(class_path, images[0])
                result = load_and_normalise(sample_path)
                
                print("Normalisation test:")
                print(f"  Image path  : {sample_path}")
                print(f"  Shape       : {result.shape}")   # Should be (224, 224, 3)
                print(f"  Min value   : {result.min():.4f}")   # Should be ~0.0
                print(f"  Max value   : {result.max():.4f}")   # Should be ~1.0
                print(f"  Data type   : {result.dtype}")   # Should be float32
                return
    
    print("No images found in data/processed/. Run resize_images.py first.")

if __name__ == "__main__":
    verify_normalisation()