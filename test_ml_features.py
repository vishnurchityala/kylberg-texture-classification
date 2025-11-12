import random
from src.preprocessing import load_dataset
from src.ml_features import extract_texture_features

X, y, classes = load_dataset(limit_per_class=1)
idx = random.randint(0, len(X) - 1)
img = X[idx]
class_name = classes[y[idx]]

print(f"Extracting integrated features for: {class_name}")

features = extract_texture_features(img)
print(f"Feature vector length: {len(features)}")