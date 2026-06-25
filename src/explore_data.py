import os

dataset_path = "data/raw/PlantVillage"

# Gather all class names and counts
classes = []
counts = []

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        classes.append(class_name)
        counts.append(count)

# Print summary statistics
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Number of classes : {len(classes)}")
print(f"Total images      : {sum(counts)}")
print(f"Average per class : {sum(counts) // len(counts)}")
print(f"Most images in    : {classes[counts.index(max(counts))]} ({max(counts)})")
print(f"Fewest images in  : {classes[counts.index(min(counts))]} ({min(counts)})")
print("=" * 60)