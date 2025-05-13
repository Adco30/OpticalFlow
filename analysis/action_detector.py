"""
Advanced motion pattern detection for the OpticalFlow system.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict


class MotionPatternDetector:
    """Detects specific motion patterns in optical flow data."""
    
    def __init__(self):
        self.motion_threshold = 0.15
        
    def analyze_flow_coherence(self, U: np.ndarray, V: np.ndarray) -> float:
        """
        Measure spatial coherence of flow vectors.
        
        Args:
            U: Horizontal flow component
            V: Vertical flow component
            
        Returns:
            Coherence score (0-1, higher means more coherent)
        """
        # Calculate gradients of U and V
        grad_u_x = cv2.Sobel(U, cv2.CV_64F, 1, 0, ksize=3)
        grad_u_y = cv2.Sobel(U, cv2.CV_64F, 0, 1, ksize=3)
        grad_v_x = cv2.Sobel(V, cv2.CV_64F, 1, 0, ksize=3)
        grad_v_y = cv2.Sobel(V, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitudes = np.sqrt(grad_u_x**2 + grad_u_y**2 + grad_v_x**2 + grad_v_y**2)
        
        # Create mask for valid flow regions
        magnitudes = np.sqrt(U**2 + V**2)
        valid_mask = magnitudes > 0
        
        if not np.any(valid_mask):
            return 0
        
        # Calculate coherence as inverse of average gradient
        avg_gradient = np.mean(gradient_magnitudes[valid_mask])
        coherence = 1 / (1 + avg_gradient)
        
        return coherence
    
    def detect_flow_boundaries(self, U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect boundaries between different flow regions.
        
        Args:
            U: Horizontal flow component
            V: Vertical flow component
            
        Returns:
            Tuple of (boundary_mask, flow_magnitude)
        """
        # Calculate flow magnitude
        magnitude = np.sqrt(U**2 + V**2)
        
        # Calculate spatial gradients of magnitude
        grad_x = cv2.Sobel(magnitude, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(magnitude, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold to find boundaries
        threshold = np.mean(gradient_magnitude) + np.std(gradient_magnitude)
        boundary_mask = (gradient_magnitude > threshold).astype(np.uint8)
        
        # Apply morphological closing to connect boundary segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary_mask = cv2.morphologyEx(boundary_mask, cv2.MORPH_CLOSE, kernel)
        
        return boundary_mask, magnitude
    
    def find_flow_regions(self, U: np.ndarray, V: np.ndarray, min_region_size: int = 100) -> List[Dict]:
        """
        Segment flow field into distinct regions.
        
        Args:
            U: Horizontal flow component
            V: Vertical flow component
            min_region_size: Minimum size for detected regions
            
        Returns:
            List of region dictionaries with properties
        """
        # Calculate flow magnitude
        magnitude = np.sqrt(U**2 + V**2)
        
        # Threshold significant motion
        motion_threshold = np.mean(magnitude) + 0.5 * np.std(magnitude)
        motion_mask = (magnitude > motion_threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask)
        
        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            # Check region size
            if stats[i, cv2.CC_STAT_AREA] < min_region_size:
                continue
                
            # Extract region properties
            mask = (labels == i)
            region_U = U[mask]
            region_V = V[mask]
            region_mag = magnitude[mask]
            
            # Calculate region statistics
            region_info = {
                'label': i,
                'area': stats[i, cv2.CC_STAT_AREA],
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                'centroid': centroids[i],
                'mean_magnitude': np.mean(region_mag),
                'std_magnitude': np.std(region_mag),
                'dominant_direction': np.arctan2(np.mean(region_V), np.mean(region_U)),
                'mask': mask
            }
            
            regions.append(region_info)
        
        return regions
    
    def track_region_motion(self, regions_t0: List[Dict], regions_t1: List[Dict], max_distance: float = 50.0) -> List[Tuple[Dict, Dict]]:
        """
        Track regions between consecutive frames.
        
        Args:
            regions_t0: Regions from first frame
            regions_t1: Regions from second frame
            max_distance: Maximum distance for matching regions
            
        Returns:
            List of (region_t0, region_t1) matches
        """
        matches = []
        
        for r0 in regions_t0:
            best_match = None
            min_distance = float('inf')
            
            for r1 in regions_t1:
                # Calculate distance between centroids
                distance = np.sqrt((r0['centroid'][0] - r1['centroid'][0])**2 + 
                                 (r0['centroid'][1] - r1['centroid'][1])**2)
                
                # Consider size similarity
                size_ratio = min(r0['area'], r1['area']) / max(r0['area'], r1['area'])
                distance_weighted = distance / (size_ratio + 0.1)
                
                if distance_weighted < min_distance and distance < max_distance:
                    min_distance = distance_weighted
                    best_match = r1
            
            if best_match is not None:
                matches.append((r0, best_match))
        
        return matches
