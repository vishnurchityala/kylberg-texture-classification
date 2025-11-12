import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

def blockwise_dct(img, block_size=8):
    """Apply 2D DCT to image in non-overlapping blocks."""
    h, w = img.shape
    dct_blocks = np.zeros_like(img, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            dct_blocks[i:i+block_size, j:j+block_size] = dct(dct(block.T, norm='ortho').T, norm='ortho')
    return dct_blocks


def blockwise_idct(dct_blocks, block_size=8):
    """Apply inverse 2D DCT to reconstruct image."""
    h, w = dct_blocks.shape
    img_recon = np.zeros_like(dct_blocks, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_blocks[i:i+block_size, j:j+block_size]
            img_recon[i:i+block_size, j:j+block_size] = idct(idct(block.T, norm='ortho').T, norm='ortho')
    return img_recon

JPEG_QUANT_MATRIX = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)


def quantize(dct_blocks, q_matrix=JPEG_QUANT_MATRIX, quality=50, block_size=8):
    """Quantize DCT coefficients block-wise."""
    h, w = dct_blocks.shape
    scale = 50 / quality if quality < 50 else 2 - quality / 50
    q_matrix_scaled = np.clip(q_matrix * scale, 1, 255)

    q_blocks = np.zeros_like(dct_blocks)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_blocks[i:i+block_size, j:j+block_size]
            q_blocks[i:i+block_size, j:j+block_size] = np.round(block / q_matrix_scaled)
    return q_blocks

def dequantize(q_blocks, q_matrix=JPEG_QUANT_MATRIX, quality=50, block_size=8):
    """Dequantize DCT coefficients block-wise."""
    h, w = q_blocks.shape
    scale = 50 / quality if quality < 50 else 2 - quality / 50
    q_matrix_scaled = np.clip(q_matrix * scale, 1, 255)

    dq_blocks = np.zeros_like(q_blocks)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = q_blocks[i:i+block_size, j:j+block_size]
            dq_blocks[i:i+block_size, j:j+block_size] = block * q_matrix_scaled
    return dq_blocks

def zigzag_scan(block):
    """Return zigzag order of 8x8 block."""
    h, w = block.shape
    result = []
    for s in range(h + w - 1):
        for i in range(max(0, s - w + 1), min(h, s + 1)):
            j = s - i
            result.append(block[i, j] if s % 2 == 0 else block[j, i])
    return np.array(result)


def inverse_zigzag(arr, block_size=8):
    """Reconstruct block from zigzag order."""
    block = np.zeros((block_size, block_size), dtype=np.float32)
    idx = 0
    for s in range(block_size * 2 - 1):
        for i in range(max(0, s - block_size + 1), min(block_size, s + 1)):
            j = s - i
            if s % 2 == 0:
                block[i, j] = arr[idx]
            else:
                block[j, i] = arr[idx]
            idx += 1
    return block

def visualize_compression(img, quality=50):
    """Demonstrate DCT-based compression and reconstruction."""
    img_gray = (img * 255).astype(np.uint8)
    img_gray = cv2.resize(img_gray, (256, 256))

    dct_blocks = blockwise_dct(img_gray)
    q_blocks = quantize(dct_blocks, quality=quality)
    dq_blocks = dequantize(q_blocks, quality=quality)
    recon_img = blockwise_idct(dq_blocks)
    recon_img = cv2.normalize(recon_img, None, 0, 1, cv2.NORM_MINMAX)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(recon_img, cmap='gray')
    plt.title(f"Reconstructed (Quality={quality})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return dct_blocks, q_blocks, recon_img
