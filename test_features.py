import random
from src.preprocessing import load_dataset
from src.features import visualize_feature_detectors

X, y, classes = load_dataset(limit_per_class=1)
print(f"Loaded {len(X)} images from {len(classes)} texture classes.")

indices = random.sample(range(len(X)), 1)

for i in indices:
    img = X[i]
    class_name = classes[y[i]]
    print(f"\nVisualizing feature detectors on: {class_name}")

    visualize_feature_detectors(img)
