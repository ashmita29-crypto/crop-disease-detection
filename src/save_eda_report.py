import os

dataset_path = "data/raw/PlantVillage"

classes = []
counts = []

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        count = len(os.listdir(class_path))
        classes.append(class_name)
        counts.append(count)

# Write a text report
report_lines = []
report_lines.append("EDA REPORT — PlantVillage Dataset")
report_lines.append("=" * 50)
report_lines.append(f"Total Classes  : {len(classes)}")
report_lines.append(f"Total Images   : {sum(counts)}")
report_lines.append(f"Avg per class  : {sum(counts)//len(counts)}")
report_lines.append(f"Max images     : {max(counts)} ({classes[counts.index(max(counts))]})")
report_lines.append(f"Min images     : {min(counts)} ({classes[counts.index(min(counts))]})")
report_lines.append("")
report_lines.append("Per-class breakdown:")
report_lines.append("-" * 50)
for cls, cnt in zip(classes, counts):
    report_lines.append(f"  {cls:<45} {cnt}")

with open("reports/eda_summary.txt", "w") as f:
    f.write("\n".join(report_lines))

print("EDA report saved to reports/eda_summary.txt")