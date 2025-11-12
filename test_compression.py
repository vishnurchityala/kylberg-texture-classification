from src.preprocessing import load_dataset
from src.compression import visualize_compression
import random

X, y, classes = load_dataset(limit_per_class=1)
idx = random.randint(0, len(X) - 1)
img = X[idx]
class_name = classes[y[idx]]

print(f"Testing compression on: {class_name}")
visualize_compression(img, quality=30)
