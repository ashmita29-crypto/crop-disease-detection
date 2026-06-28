import os

# Path to the raw dataset
dataset_path = r"data\raw\PlantVillage"

# List all class folders
classes = sorted(os.listdir(dataset_path))

print(f"Total number of classes: {len(classes)}")
print("\nAll classes:")
for i, cls in enumerate(classes):
    print(f"  {i+1}. {cls}")