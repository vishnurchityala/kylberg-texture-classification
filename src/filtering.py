import cv2
import numpy as np
import matplotlib.pyplot as plt


def mean_filter(img, ksize=5):
    """Apply mean (average) filtering."""
    return cv2.blur(img, (ksize, ksize))


def median_filter(img, ksize=5):
    """Apply median filtering."""
    img_uint8 = (img * 255).astype(np.uint8)
    return cv2.medianBlur(img_uint8, ksize) / 255.0


def gaussian_filter(img, ksize=5, sigma=1):
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def sobel_edge(img):
    """Sobel edge detection (combines x and y)."""
    img_uint8 = (img * 255).astype(np.uint8)
    sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return (sobel / sobel.max()) if sobel.max() != 0 else sobel


def laplacian_edge(img):
    """Laplacian edge detection."""
    img_uint8 = (img * 255).astype(np.uint8)
    lap = cv2.Laplacian(img_uint8, cv2.CV_64F)
    lap = np.absolute(lap)
    return (lap / lap.max()) if lap.max() != 0 else lap


def laplacian_of_gaussian(img, ksize=5, sigma=1):
    """Apply Laplacian of Gaussian (LoG) filter."""
    img_uint8 = (img * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img_uint8, (ksize, ksize), sigma)
    log = cv2.Laplacian(blur, cv2.CV_64F)
    log = np.absolute(log)
    return (log / log.max()) if log.max() != 0 else log


def visualize_filters(img):
    """Show outputs of common filters."""
    filters = {
        "Original": img,
        "Mean": mean_filter(img),
        "Median": median_filter(img),
        "Gaussian": gaussian_filter(img),
        "Sobel Edge": sobel_edge(img),
        "Laplacian Edge": laplacian_edge(img),
        "LoG": laplacian_of_gaussian(img)
    }

    plt.figure(figsize=(12, 8))
    for i, (name, fimg) in enumerate(filters.items()):
        plt.subplot(2, 4, i + 1)
        plt.imshow(fimg, cmap="gray")
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
