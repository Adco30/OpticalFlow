"""
Core optical flow algorithms for the OpticalFlow visualization system.
Contains implementations of Lucas-Kanade and hierarchical optical flow algorithms.
"""
import cv2
import numpy as np
from typing import Tuple, Optional

try:
    from ..utils.config import FlowConfiguration
except ImportError:
    # For direct execution during development
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.config import FlowConfiguration


def calculate_spatial_gradients(image: np.ndarray, gradient_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spatial gradients of an image.
    
    Args:
        image: Input image
        gradient_scale: Scaling factor for gradients
        
    Returns:
        Tuple of (Ix, Iy) spatial gradients
    """
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=gradient_scale, borderType=cv2.BORDER_DEFAULT)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=gradient_scale, borderType=cv2.BORDER_DEFAULT)
    return Ix, Iy


def calculate_temporal_gradient(image1: np.ndarray, image2: np.ndarray, temporal_kernel_size: int) -> np.ndarray:
    """
    Calculate temporal gradient between two images.
    
    Args:
        image1: First image
        image2: Second image
        temporal_kernel_size: Size of temporal kernel
        
    Returns:
        Temporal gradient It
    """
    kernel_size = int(temporal_kernel_size)
    It = cv2.filter2D(
        image2.astype(np.float32) - image1.astype(np.float32), 
        -1,
        np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    )
    return It


def compute_flow_derivatives(img1: np.ndarray, img2: np.ndarray, config: FlowConfiguration) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spatial and temporal derivatives for optical flow calculation.
    
    Args:
        img1: First image
        img2: Second image
        config: Flow configuration object
        
    Returns:
        Tuple of (Ix, Iy, It) derivatives
    """
    Ix, Iy = calculate_spatial_gradients(img1, config.gradient_scale)
    It = calculate_temporal_gradient(img1, img2, config.temporal_kernel_size)
    return Ix, Iy, It


def basic_lucas_kanade_flow(img1: np.ndarray, img2: np.ndarray, window_size: int, kernel_type: str, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow using the Lucas-Kanade method.
    
    Args:
        img1: First image
        img2: Second image
        window_size: Window size for averaging
        kernel_type: Type of kernel ('uniform' or 'gaussian')
        sigma: Sigma value for Gaussian kernel
        
    Returns:
        Tuple of (U, V) flow components
    """
    window_size = int(window_size)
    
    # Create configuration object for derivatives
    config = FlowConfiguration()
    
    # Compute spatial and temporal derivatives
    Ix, Iy, It = compute_flow_derivatives(img1, img2, config)
    
    # Create kernel based on type
    if kernel_type == 'uniform':
        kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    else:
        # Gaussian kernel
        kernel_1d = cv2.getGaussianKernel(window_size, sigma)
        kernel = kernel_1d @ kernel_1d.T
    
    # Apply windowing to compute sums of products
    Ixx = cv2.filter2D(Ix * Ix, -1, kernel)
    Ixy = cv2.filter2D(Ix * Iy, -1, kernel)
    Iyy = cv2.filter2D(Iy * Iy, -1, kernel)
    Ixt = cv2.filter2D(Ix * It, -1, kernel)
    Iyt = cv2.filter2D(Iy * It, -1, kernel)
    
    # Initialize flow fields
    U = np.zeros_like(img1, dtype=np.float32)
    V = np.zeros_like(img1, dtype=np.float32)
    
    # Compute flow only where determinant is not too small (well-conditioned)
    det = Ixx * Iyy - Ixy * Ixy
    valid_points = det > 1e-6
    
    # Solve the 2x2 linear system for valid points
    U[valid_points] = (Iyy[valid_points] * (-Ixt[valid_points]) - 
                      Ixy[valid_points] * (-Iyt[valid_points])) / det[valid_points]
    V[valid_points] = (-Ixy[valid_points] * (-Ixt[valid_points]) + 
                      Ixx[valid_points] * (-Iyt[valid_points])) / det[valid_points]
    
    return U, V


def reduce_image_scale(image: np.ndarray) -> np.ndarray:
    """
    Reduces an image to half its dimensions.
    
    Args:
        image: Input image
        
    Returns:
        Reduced image
    """
    # 5-tap separable filter
    kernel = np.array([1, 4, 6, 4, 1]) / 16.0

    # Apply separable filter
    temp = cv2.filter2D(image, -1, kernel.reshape(1, -1))
    reduced = cv2.filter2D(temp, -1, kernel.reshape(-1, 1))

    # Subsample
    return reduced[::2, ::2]


def create_gaussian_pyramid(image: np.ndarray, levels: int) -> list:
    """
    Creates a Gaussian pyramid of images.
    
    Args:
        image: Input image
        levels: Number of pyramid levels
        
    Returns:
        List of images forming the pyramid
    """
    pyramid = [image.copy()]
    for _ in range(levels - 1):
        pyramid.append(reduce_image_scale(pyramid[-1]))
    return pyramid


def expand_image_scale(image: np.ndarray) -> np.ndarray:
    """
    Expands an image by doubling its width and height.
    
    Args:
        image: Input image
        
    Returns:
        Expanded image
    """
    h, w = image.shape
    expanded = np.zeros((h * 2, w * 2), dtype=image.dtype)
    expanded[::2, ::2] = image

    # 5-tap separable filter
    kernel = np.array([1, 4, 6, 4, 1]) / 16.0

    # Apply separable filter
    temp = cv2.filter2D(expanded, -1, kernel.reshape(1, -1))
    return cv2.filter2D(temp, -1, kernel.reshape(-1, 1))


def warp_image(image: np.ndarray, U: np.ndarray, V: np.ndarray, interpolation: int, border_mode: int) -> np.ndarray:
    """
    Warps an image using optical flow displacement vectors.
    
    Args:
        image: Image to warp
        U: Horizontal flow component
        V: Vertical flow component
        interpolation: Interpolation method
        border_mode: Border mode for warping
        
    Returns:
        Warped image
    """
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (x + U).astype(np.float32)
    map_y = (y + V).astype(np.float32)

    return cv2.remap(image, map_x, map_y, interpolation, borderMode=border_mode)


def hierarchical_lucas_kanade_flow(img1: np.ndarray, img2: np.ndarray, levels: int, window_size: int, 
                               kernel_type: str, sigma: float, interpolation: int, border_mode: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes optical flow using Hierarchical Lucas-Kanade algorithm.
    
    Args:
        img1: First image
        img2: Second image
        levels: Number of pyramid levels
        window_size: Window size for averaging
        kernel_type: Type of kernel ('uniform' or 'gaussian')
        sigma: Sigma value for Gaussian kernel
        interpolation: Interpolation method for warping
        border_mode: Border mode for warping
        
    Returns:
        Tuple of (U, V) flow components
    """
    # Build pyramids
    pyramid1 = create_gaussian_pyramid(img1, levels)
    pyramid2 = create_gaussian_pyramid(img2, levels)

    # Initialize flow at coarsest level
    U = np.zeros_like(pyramid1[-1], dtype=np.float32)
    V = np.zeros_like(pyramid1[-1], dtype=np.float32)

    # Process each level from coarse to fine
    for level in range(levels - 1, -1, -1):
        # Expand flow to next level if not at finest level
        if level < levels - 1:
            U = 2 * expand_image_scale(U)
            V = 2 * expand_image_scale(V)
            # Adjust size if needed
            U = U[:pyramid1[level].shape[0], :pyramid1[level].shape[1]]
            V = V[:pyramid1[level].shape[0], :pyramid1[level].shape[1]]

        # Warp image
        warped = warp_image(pyramid2[level], U, V, interpolation, border_mode)

        # Compute flow update
        dU, dV = basic_lucas_kanade_flow(pyramid1[level], warped, window_size, kernel_type, sigma)

        # Update flow
        U += dU
        V += dV

    return U, V
