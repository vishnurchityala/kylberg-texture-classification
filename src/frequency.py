import numpy as np
import cv2
import matplotlib.pyplot as plt

def fourier_transform(img):
    """Compute 2D Fourier Transform and its magnitude spectrum."""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return fshift, magnitude_spectrum


def inverse_fourier_transform(fshift):
    """Compute inverse Fourier Transform to get back spatial image."""
    f_ishift = np.fft.ifftshift(fshift)
    img_recon = np.fft.ifft2(f_ishift)
    img_recon = np.abs(img_recon)
    img_recon = cv2.normalize(img_recon, None, 0, 1, cv2.NORM_MINMAX)
    return img_recon

def ideal_lowpass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 1, -1)
    return mask


def ideal_highpass_filter(shape, cutoff):
    return 1 - ideal_lowpass_filter(shape, cutoff)


def gaussian_lowpass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    mask = np.exp(-((X - ccol)**2 + (Y - crow)**2) / (2 * cutoff**2))
    return mask


def gaussian_highpass_filter(shape, cutoff):
    return 1 - gaussian_lowpass_filter(shape, cutoff)


def butterworth_lowpass_filter(shape, cutoff, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    mask = 1 / (1 + (distance / cutoff)**(2 * order))
    return mask


def butterworth_highpass_filter(shape, cutoff, order=2):
    return 1 - butterworth_lowpass_filter(shape, cutoff, order)


def apply_frequency_filter(img, filter_mask):
    fshift, mag = fourier_transform(img)
    filtered_shift = fshift * filter_mask
    filtered_img = inverse_fourier_transform(filtered_shift)
    return filtered_img, mag, np.log(np.abs(filtered_shift) + 1)


def visualize_frequency_filters(img):
    """Visualize Fourier filters and results."""
    filters = {
        "Ideal Low-pass": ideal_lowpass_filter(img.shape, 30),
        "Ideal High-pass": ideal_highpass_filter(img.shape, 30),
        "Gaussian Low-pass": gaussian_lowpass_filter(img.shape, 30),
        "Gaussian High-pass": gaussian_highpass_filter(img.shape, 30),
        "Butterworth Low-pass": butterworth_lowpass_filter(img.shape, 30),
        "Butterworth High-pass": butterworth_highpass_filter(img.shape, 30),
    }

    plt.figure(figsize=(14, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    i = 2
    for name, mask in filters.items():
        filtered_img, mag, _ = apply_frequency_filter(img, mask)
        plt.subplot(3, 3, i)
        plt.imshow(filtered_img, cmap='gray')
        plt.title(name)
        plt.axis('off')
        i += 1

    plt.tight_layout()
    plt.show()
