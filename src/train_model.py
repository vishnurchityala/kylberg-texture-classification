import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import load_dataset
from src.ml_features import extract_texture_features

LIMIT_PER_CLASS = 160
SAVE = True
N_COMPONENTS = 50

def build_feature_dataset(limit_per_class=LIMIT_PER_CLASS):
    X, y, classes = load_dataset(limit_per_class=limit_per_class)
    features = [extract_texture_features(img) for img in X]
    return np.array(features), np.array(y), classes


def train_and_evaluate(save=SAVE):
    print("Extracting features...")
    X_features, y, classes = build_feature_dataset(limit_per_class=LIMIT_PER_CLASS)
    print(f"Dataset built with {len(X_features)} samples, {X_features.shape[1]} features each")

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Applying PCA...")
    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("Training SVM model...")
    model = SVC(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train_pca, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_pca)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/svm_model.pkl")
        joblib.dump(pca, "models/pca.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        print("\nModel, PCA, and Scaler saved in 'models/' directory.")
