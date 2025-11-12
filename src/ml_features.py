import cv2
import numpy as np
from skimage.feature import hog
from src.features import compute_glcm_features
from src.filtering import mean_filter, gaussian_filter, sobel_edge
from src.frequency import fourier_transform
from scipy.stats import kurtosis, skew


def extract_sift_features(img):
    """Extract SIFT descriptors and return mean pooled vector."""
    img_uint8 = (img * 255).astype(np.uint8)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_uint8, None)

    if descriptors is None:
        return np.zeros(128)
    return np.mean(descriptors, axis=0)

def extract_hog_features(img):
    """Extract Histogram of Oriented Gradients (HOG) features."""
    img_resized = cv2.resize(img, (128, 128))
    features, hog_image = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )
    return features, hog_image

def extract_spatial_features(img):
    """Compute mean, variance, entropy, and filtered responses."""
    mean_val = np.mean(img)
    std_val = np.std(img)
    energy = np.sum(img ** 2) / img.size

    smooth = mean_filter(img)
    edges = sobel_edge(img)

    smooth_mean = np.mean(smooth)
    edge_mean = np.mean(edges)
    edge_energy = np.sum(edges ** 2) / edges.size

    return np.array([mean_val, std_val, energy, smooth_mean, edge_mean, edge_energy])

def extract_frequency_features(img):
    """Extract magnitude spectrum statistics from FFT."""
    _, mag = fourier_transform(img)
    return np.array([
        np.mean(mag),
        np.std(mag),
        skew(mag.ravel()),
        kurtosis(mag.ravel())
    ])

def extract_texture_features(img):
    """Combine spatial, GLCM, frequency, SIFT, and HOG into one vector."""
    img = cv2.resize(img, (256, 256))
    img = (img - img.min()) / (img.max() - img.min())

    spatial_feats = extract_spatial_features(img)
    glcm_feats = list(compute_glcm_features(img).values())
    freq_feats = extract_frequency_features(img)
    sift_feats = extract_sift_features(img)
    hog_feats, _ = extract_hog_features(img)

    final_features = np.concatenate([
        spatial_feats,
        glcm_feats,
        freq_feats,
        sift_feats,
        hog_feats
    ])
    return final_features
