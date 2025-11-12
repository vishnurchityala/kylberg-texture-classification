import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.preprocessing import load_dataset
from src.ml_features import extract_texture_features

MODEL_PATH = "models/svm_model.pkl"
PCA_PATH = "models/pca.pkl"
SCALER_PATH = "models/scaler.pkl"
RESULTS_DIR = "results"
LIMIT_PER_CLASS = 160

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_model_components():
    if not all(os.path.exists(p) for p in [MODEL_PATH, PCA_PATH, SCALER_PATH]):
        raise FileNotFoundError("Missing model, PCA, or scaler file in 'models/' directory.")
    model = joblib.load(MODEL_PATH)
    pca = joblib.load(PCA_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, pca, scaler

def evaluate_model(limit_per_class=LIMIT_PER_CLASS):
    ensure_dir(RESULTS_DIR)

    print("Loading dataset...")
    X, y, classes = load_dataset(limit_per_class=limit_per_class)

    print("Extracting features...")
    features = [extract_texture_features(img) for img in X]
    X_features = np.array(features)

    print("Loading trained model components...")
    model, pca, scaler = load_model_components()

    print("Transforming features...")
    X_scaled = scaler.transform(X_features)
    X_pca = pca.transform(X_scaled)

    print("Evaluating model performance...")
    y_pred = model.predict(X_pca)

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=classes)
    cm = confusion_matrix(y, y_pred)

    print(f"\nOverall Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    # Save text results
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, cm, fmt="%d")

    # Save confusion matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", annot=False, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"\nResults saved to '{RESULTS_DIR}/' folder.")
    return acc

if __name__ == "__main__":
    evaluate_model()