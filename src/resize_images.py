import os
from PIL import Image

SOURCE_DIR = "data/raw/PlantVillage"
DEST_DIR = "data/processed"
TARGET_SIZE = (224, 224)

def resize_all_images():
    classes = sorted(os.listdir(SOURCE_DIR))
    total_processed = 0
    
    for class_name in classes:
        source_class_path = os.path.join(SOURCE_DIR, class_name)
        dest_class_path = os.path.join(DEST_DIR, class_name)
        
        if not os.path.isdir(source_class_path):
            continue
        
        # Create the destination class folder
        os.makedirs(dest_class_path, exist_ok=True)
        
        # Process every image in this class
        for image_file in os.listdir(source_class_path):
            source_path = os.path.join(source_class_path, image_file)
            dest_path = os.path.join(dest_class_path, image_file)
            
            try:
                img = Image.open(source_path)
                img = img.resize(TARGET_SIZE)         # Resize to 224x224
                img = img.convert("RGB")              # Ensure 3 colour channels
                img.save(dest_path)
                total_processed += 1
            except Exception as e:
                print(f"Error with {image_file}: {e}")
        
        print(f"Resized class: {class_name}")
    
    print(f"\nDone. Total images resized: {total_processed}")

if __name__ == "__main__":
    resize_all_images()