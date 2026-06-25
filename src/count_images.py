import os

dataset_path = "data/raw/PlantVillage"

print(f"{'Class Name':<50} {'Image Count':>12}")
print("-" * 65)

total = 0
class_counts = {}

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)
    
    # Only count if it's a folder
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        class_counts[class_name] = count
        total += count
        print(f"{class_name:<50} {count:>12}")

print("-" * 65)
print(f"{'TOTAL IMAGES':<50} {total:>12}")