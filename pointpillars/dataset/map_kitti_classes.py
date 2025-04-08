import os

CLASS_MAP = {
    "car": "Car",
    "truck": "Car",
    "bus": "Car",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
    "pedestrian": "Pedestrian"
}

kitti_label_dir = './kitti_sample/training/label_2'

if os.path.exists(kitti_label_dir):
    print(f"Directory exists: {kitti_label_dir}")

for label_file in os.listdir(kitti_label_dir):
    with open(os.path.join(kitti_label_dir, label_file), 'r+') as f:
        lines = []
        for line in f:
            parts = line.strip().split()
            original_class = parts[0]
            if original_class in CLASS_MAP:
                parts[0] = CLASS_MAP[original_class]
                lines.append(" ".join(parts) + "\n")
        f.seek(0)
        f.writelines(lines)
        f.truncate()