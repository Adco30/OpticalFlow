"""
Image preprocessing and manipulation operations for the OpticalFlow system.
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

try:
    from ..utils.config import FlowConfiguration
except ImportError:
    # For direct execution during development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.config import FlowConfiguration


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to an image.
    
    Args:
        image: Input image
        gamma: Gamma value
        
    Returns:
        Gamma corrected image
    """
    return np.power(image, 1/gamma)


def apply_bilateral_filter(image: np.ndarray, config: FlowConfiguration) -> np.ndarray:
    """
    Apply bilateral filter to an image.
    
    Args:
        image: Input image
        config: Flow configuration containing filter parameters
        
    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(
        image, 
        config.bilateral_d, 
        config.bilateral_sigma_color, 
        config.bilateral_sigma_space
    )


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Input image
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (0, 0), sigma)


def preprocess_image(image: np.ndarray, config: FlowConfiguration) -> np.ndarray:
    """
    Preprocess image with gamma correction and filtering.
    
    Args:
        image: Input image
        config: Flow configuration
        
    Returns:
        Preprocessed image
    """
    # Apply gamma correction
    gamma_corrected = gamma_correction(image, config.gamma)
    
    # Apply filtering
    if config.use_bilateral:
        filtered = apply_bilateral_filter(gamma_corrected, config)
    else:
        filtered = apply_gaussian_blur(gamma_corrected, config.gaussian_sigma)
    
    return filtered


def normalize_image(image: np.ndarray, scale_range: Tuple[float, float] = (0, 255)) -> np.ndarray:
    """
    Normalize image values to a specified range.
    
    Args:
        image: Input image
        scale_range: Target range (min, max)
        
    Returns:
        Normalized image
    """
    image_out = np.zeros(image.shape)
    cv2.normalize(image, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)
    return image_out


def compute_image_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute normalized difference between two images.
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        Normalized difference image
    """
    diff = np.abs(img2 - img1)
    if diff.max() > 0:
        diff = (diff * 255.0 / diff.max()).astype(np.uint8)
    return diff


def load_test_images(base_path: str) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    Load and preprocess images for optical flow computation.
    
    Args:
        base_path: Path to image directory
        
    Returns:
        Tuple of (base_image, list of (filename, image) pairs)
    """
    base_img = cv2.imread(os.path.join(base_path, 'Shift0.png'), cv2.IMREAD_GRAYSCALE)
    base_img = base_img.astype(np.float32) / 255.0

    other_images = []
    image_order = ['ShiftR2.png', 'ShiftR5U5.png', 'ShiftR10.png',
                   'ShiftR20.png', 'ShiftR40.png']

    for filename in image_order:
        img_path = os.path.join(base_path, filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            other_images.append((filename, img))

    return base_img, other_images


def combine_pyramid_images(pyramid_images: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Stacks pyramid images side-by-side for visualization.
    
    Args:
        pyramid_images: List of pyramid level images
        
    Returns:
        Combined image or None if input is empty
    """
    if not pyramid_images:
        return None

    # Normalize each image
    normalized_images = [normalize_image(img) for img in pyramid_images]

    # Calculate dimensions
    max_height = max(img.shape[0] for img in normalized_images)
    total_width = sum(img.shape[1] for img in normalized_images)

    # Create output image
    combined = np.zeros((max_height, total_width), dtype=np.uint8)

    # Stack images
    x_offset = 0
    for img in normalized_images:
        h, w = img.shape
        combined[:h, x_offset:x_offset + w] = img.astype(np.uint8)
        x_offset += w

    return combined
