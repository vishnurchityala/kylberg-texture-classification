from src.preprocessing import load_dataset
from src.filtering import visualize_filters

X, y, classes = load_dataset(limit_per_class=1)

print(f"Loaded {len(X)} images from {len(classes)} texture classes.")

sample = X[0]
print(f"Displaying filters for class: {classes[y[0]]}")

visualize_filters(sample)