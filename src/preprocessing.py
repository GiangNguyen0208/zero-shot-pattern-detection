"""Image preprocessing for technical drawing pattern detection."""

import cv2
import numpy as np


def load_image(image_input):
    """Load image from file path or numpy array."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot load image: {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def to_grayscale(img):
    """Convert image to grayscale if needed."""
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def enhance_contrast(img, clip_limit=2.0, tile_size=8):
    """Apply CLAHE contrast enhancement for better feature visibility."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
    )
    return clahe.apply(img)


def adaptive_binarize(img, block_size=15, c=5):
    """Adaptive thresholding for handling uneven lighting in scanned drawings."""
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def denoise(img, strength=5):
    """Light denoising to clean up scan artifacts."""
    return cv2.fastNlMeansDenoising(img, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def extract_edges(img, low=50, high=150):
    """Canny edge extraction for edge-based matching."""
    return cv2.Canny(img, low, high)


def preprocess_image(img, use_edges=False, denoise_strength=0):
    """Full preprocessing pipeline for drawing images.

    Returns grayscale image ready for matching.
    """
    gray = to_grayscale(img)

    if denoise_strength > 0:
        gray = denoise(gray, strength=denoise_strength)

    enhanced = enhance_contrast(gray)

    if use_edges:
        return extract_edges(enhanced)

    return enhanced


def preprocess_pattern(img, use_edges=False):
    """Preprocess pattern image for matching."""
    gray = to_grayscale(img)
    enhanced = enhance_contrast(gray)

    if use_edges:
        return extract_edges(enhanced)

    return enhanced


def build_image_pyramid(img, num_levels=3, scale_factor=0.5):
    """Build Gaussian image pyramid for coarse-to-fine matching."""
    pyramid = [img]
    current = img
    for _ in range(num_levels - 1):
        h, w = current.shape[:2]
        new_h, new_w = max(1, int(h * scale_factor)), max(1, int(w * scale_factor))
        current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pyramid.append(current)
    return pyramid
