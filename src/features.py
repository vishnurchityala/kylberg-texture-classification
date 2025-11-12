import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edge(img, low_thresh=100, high_thresh=200):
    """Apply Canny edge detector."""
    img_uint8 = (img * 255).astype(np.uint8)
    edges = cv2.Canny(img_uint8, low_thresh, high_thresh)
    return edges / 255.0


def harris_corners(img, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """Detect corners using Harris method."""
    img_gray = (img * 255).astype(np.uint8)
    img_float = np.float32(img_gray)
    dst = cv2.cornerHarris(img_float, block_size, ksize, k)
    dst = cv2.dilate(dst, None)

    corners = dst > threshold * dst.max()

    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    vis[corners] = [255, 0, 0]

    return vis / 255.0, corners

def morphological_operations(img, kernel_size=5):
    """Apply dilation, erosion, opening, and closing."""
    img_uint8 = (img * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilation = cv2.dilate(img_uint8, kernel, iterations=1)
    erosion = cv2.erode(img_uint8, kernel, iterations=1)
    opening = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel)

    return dilation, erosion, opening, closing


def boundary_detection(img, kernel_size=3):
    """Detect object boundaries using morphological gradient."""
    img_uint8 = (img * 255).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gradient = cv2.morphologyEx(img_uint8, cv2.MORPH_GRADIENT, kernel)
    return gradient / 255.0


def hole_filling(binary_img):
    """Fill holes in binary images."""
    img_uint8 = (binary_img * 255).astype(np.uint8)
    h, w = img_uint8.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = img_uint8.copy()

    cv2.floodFill(filled, mask, (0, 0), 255)
    filled_inv = cv2.bitwise_not(filled)
    filled_result = img_uint8 | filled_inv
    return filled_result / 255.0

from skimage.feature import graycomatrix, graycoprops

def compute_glcm_features(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Compute GLCM texture features."""
    img_uint8 = (img * 255).astype(np.uint8)

    glcm = graycomatrix(img_uint8, distances=distances, angles=angles, symmetric=True, normed=True)

    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'ASM': graycoprops(glcm, 'ASM').mean()
    }
    return features

def visualize_feature_detectors(img):
    """Visualize edge, corner, morphological, and texture features."""
    canny = canny_edge(img)
    harris_vis, _ = harris_corners(img)

    dilation, erosion, opening, closing = morphological_operations(img)
    boundary = boundary_detection(img)

    binary = (img > 0.5).astype(np.float32)
    filled = hole_filling(binary)

    glcm_feats = compute_glcm_features(img)

    plt.figure(figsize=(14, 10))

    imgs = [
        (img, "Original"),
        (canny, "Canny Edge"),
        (harris_vis, "Harris Corners"),
        (dilation / 255.0, "Dilation"),
        (erosion / 255.0, "Erosion"),
        (opening / 255.0, "Opening"),
        (closing / 255.0, "Closing"),
        (boundary, "Boundary"),
        (filled, "Hole Filled"),
    ]

    for i, (im, title) in enumerate(imgs, 1):
        plt.subplot(3, 3, i)
        plt.imshow(im, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("GLCM Texture Features:")
    for key, val in glcm_feats.items():
        print(f"   {key:<15}: {val:.4f}")
