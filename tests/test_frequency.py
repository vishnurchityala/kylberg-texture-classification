from src.preprocessing import load_dataset
from src.frequency import visualize_frequency_filters

X, y, classes = load_dataset(limit_per_class=1)
img = X[0]
print(f"Loaded sample from: {classes[y[0]]}")

visualize_frequency_filters(img)
