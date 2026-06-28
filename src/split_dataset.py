"""
split_dataset.py

Splits the processed images into train, validation, and test sets.
Ratio: 70% train / 15% validation / 15% test

The result:
  data/splits/train/<class_name>/<images>
  data/splits/val/<class_name>/<images>
  data/splits/test/<class_name>/<images>
"""

import os
import shutil
import random

PROCESSED_DIR = "data/processed"
SPLITS_DIR = "data/splits"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # Setting a seed makes the split reproducible

def split_dataset():
    # Create split folders
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(SPLITS_DIR, split), exist_ok=True)
    
    class_names = sorted(os.listdir(PROCESSED_DIR))
    
    print("Splitting dataset...")
    print(f"  Train : {int(TRAIN_RATIO*100)}%")
    print(f"  Val   : {int(VAL_RATIO*100)}%")
    print(f"  Test  : {int(TEST_RATIO*100)}%")
    print()
    
    for class_name in class_names:
        class_path = os.path.join(PROCESSED_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        
        images = os.listdir(class_path)
        random.shuffle(images)
        
        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        
        train_images = images[:n_train]
        val_images   = images[n_train : n_train + n_val]
        test_images  = images[n_train + n_val:]
        
        for split, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
            dest_folder = os.path.join(SPLITS_DIR, split, class_name)
            os.makedirs(dest_folder, exist_ok=True)
            for img_file in split_images:
                src = os.path.join(class_path, img_file)
                dst = os.path.join(dest_folder, img_file)
                shutil.copy2(src, dst)
        
        print(f"  {class_name}: {n_train} train, {n_val} val, {len(test_images)} test")
    
    print("\nSplit complete!")

if __name__ == "__main__":
    split_dataset()