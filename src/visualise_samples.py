import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import random

dataset_path = "data/raw/PlantVillage"

# Pick 9 random classes to display
all_classes = sorted(os.listdir(dataset_path))
selected_classes = random.sample(all_classes, 9)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Sample Images from PlantVillage Dataset", fontsize=16, y=1.02)

for ax, class_name in zip(axes.flatten(), selected_classes):
    class_path = os.path.join(dataset_path, class_name)
    images = os.listdir(class_path)
    
    # Pick one random image from this class
    random_image = random.choice(images)
    image_path = os.path.join(class_path, random_image)
    
    img = Image.open(image_path)
    ax.imshow(img)
    ax.set_title(class_name[:30], fontsize=9, wrap=True)
    ax.axis("off")

plt.tight_layout()
plt.savefig("reports/sample_images.png", dpi=150)
print("Sample image grid saved to reports/sample_images.png")