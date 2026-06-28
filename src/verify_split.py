"""
verify_split.py

Checks that the split folders exist and prints how many images
are in each split for each class. Verifies everything looks correct.
"""

import os

SPLITS_DIR = "data/splits"

for split in ["train", "val", "test"]:
    split_path = os.path.join(SPLITS_DIR, split)
    if not os.path.exists(split_path):
        print(f"ERROR: {split_path} does not exist. Run split_dataset.py first.")
        continue
    
    classes = sorted(os.listdir(split_path))
    total = 0
    
    print(f"\n{'='*50}")
    print(f"SPLIT: {split.upper()}")
    print(f"{'='*50}")
    
    for class_name in classes:
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            total += count
            print(f"  {class_name:<45} {count:>6} images")
    
    print(f"  {'TOTAL':<45} {total:>6} images")