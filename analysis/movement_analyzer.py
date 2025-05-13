"""
Movement analysis tools for the OpticalFlow system.
Provides utilities for analyzing optical flow patterns and detecting movement blocks.
"""
import cv2
import numpy as np
from typing import Tuple, Optional


class MotionAnalyzer:
    """Analyzes motion patterns in optical flow data."""
    
    def detect_moving_region(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.1, 
                           min_region_size: int = 20) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect bounding box of moving region between two images.
        
        Args:
            img1: First image
            img2: Second image
            threshold: Threshold for difference detection
            min_region_size: Minimum size of moving region
            
        Returns:
            Tuple of (x, y, width, height) or None if no region detected
        """
        # Calculate absolute difference
        diff = np.abs(img2 - img1)
        
        # Normalize and threshold the difference
        if diff.max() > 0:
            diff_norm = diff / diff.max()
            mask = (diff_norm > threshold).astype(np.uint8) * 255
        else:
            return None
            
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest contour (assuming it's the moving region)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Filter out very small detections
        if w < min_region_size or h < min_region_size:
            return None
            
        return (x, y, w, h)
    
    def calculate_motion_metrics(self, U: np.ndarray, V: np.ndarray, region_rect: Optional[Tuple[int, int, int, int]] = None) -> dict:
        """
        Calculate various motion metrics from flow fields.
        
        Args:
            U: Horizontal flow component
            V: Vertical flow component
            region_rect: Optional bounding box to focus analysis
            
        Returns:
            Dictionary of motion metrics
        """
        # Calculate magnitudes
        magnitudes = np.sqrt(U**2 + V**2)
        
        # Create masks for inside/outside region
        if region_rect is not None:
            x, y, w, h = region_rect
            inside_mask = np.zeros_like(U, dtype=bool)
            inside_mask[y:y+h, x:x+w] = True
            outside_mask = ~inside_mask
        else:
            inside_mask = np.ones_like(U, dtype=bool)
            outside_mask = np.zeros_like(U, dtype=bool)
        
        # Calculate metrics
        metrics = {
            'mean_magnitude': np.mean(magnitudes[inside_mask]) if np.any(inside_mask) else 0,
            'max_magnitude': np.max(magnitudes[inside_mask]) if np.any(inside_mask) else 0,
            'avg_direction': None,
            'motion_coverage': None,
            'outlier_ratio': None
        }
        
        # Calculate average direction in region
        if np.any(inside_mask) and metrics['mean_magnitude'] > 0:
            U_reg = U[inside_mask]
            V_reg = V[inside_mask]
            # Compute mean direction using complex representation
            directions = np.arctan2(V_reg, U_reg)
            metrics['avg_direction'] = np.angle(np.mean(np.exp(1j * directions)))
        
        # Calculate motion coverage (fraction of pixels with significant motion)
        if np.any(inside_mask):
            motion_threshold = 0.1 * metrics['mean_magnitude']
            motion_pixels = np.sum((magnitudes[inside_mask] > motion_threshold))
            total_pixels = np.sum(inside_mask)
            metrics['motion_coverage'] = motion_pixels / total_pixels if total_pixels > 0 else 0
        
        # Calculate outlier ratio (motion outside region vs inside)
        if region_rect is not None and np.any(inside_mask) and np.any(outside_mask):
            inside_mean = np.mean(magnitudes[inside_mask])
            outside_mean = np.mean(magnitudes[outside_mask])
            metrics['outlier_ratio'] = outside_mean / inside_mean if inside_mean > 0 else 1.0
        
        return metrics
