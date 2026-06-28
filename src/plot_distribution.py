import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no popup window)
import matplotlib.pyplot as plt

dataset_path = "data/raw/PlantVillage"

classes = []
counts = []

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        classes.append(class_name)
        counts.append(count)

# Create a horizontal bar chart (easier to read with long names)
plt.figure(figsize=(14, 16))
plt.barh(classes, counts, color='steelblue', edgecolor='white')
plt.xlabel("Number of Images", fontsize=13)
plt.title("PlantVillage Dataset — Images Per Class", fontsize=15, pad=20)
plt.tight_layout()

# Save to reports folder
plt.savefig("reports/class_distribution.png", dpi=150)
print("Chart saved to reports/class_distribution.png")