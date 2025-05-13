"""
Video loading and flow visualization utilities for the OpticalFlow system.
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
from PyQt5.QtCore import QObject, pyqtSignal

from utils.config import FlowConfiguration


def load_video_frames(video_path: str, display: bool = False) -> List[np.ndarray]:
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to video file
        display: Whether to display frames while loading
        
    Returns:
        List of video frames
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        if display and cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    return frames


class FlowVisualizationGenerator:
    """Generates visualizations for optical flow data."""
    
    @staticmethod
    def create_flow_visualization(U: np.ndarray, V: np.ndarray, image_shape: Tuple[int, int], 
                                config: FlowConfiguration, region_rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Create a visual representation of optical flow vectors.
        
        Args:
            U: Horizontal flow component
            V: Vertical flow component
            image_shape: Original image dimensions
            config: Flow configuration for visualization parameters
            region_rect: Optional bounding box for motion region
            
        Returns:
            Flow visualization bitmap
        """
        h, w = image_shape
        scale_factor = 2
        scaled_h, scaled_w = h * scale_factor, w * scale_factor
        
        # Create white background
        visualization = np.full((scaled_h, scaled_w, 3), 255, dtype=np.uint8)
        
        # Draw motion region if provided
        if region_rect:
            x, y, width, height = region_rect
            x, y = x * scale_factor, y * scale_factor
            width, height = width * scale_factor, height * scale_factor
            
            # Semi-transparent region overlay
            overlay = visualization.copy()
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (200, 200, 250), -1)
            alpha = 0.3
            visualization = cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0)
        
        # Calculate flow statistics
        magnitudes = np.sqrt(U**2 + V**2)
        avg_magnitude = np.mean(magnitudes[magnitudes > 0]) if np.any(magnitudes > 0) else 1.0
        
        # Draw flow vectors
        step = int(getattr(config, 'grid_spacing', 10)) * scale_factor
        
        for j in range(0, scaled_h, step):
            for i in range(0, scaled_w, step):
                orig_j, orig_i = min(h-1, j // scale_factor), min(w-1, i // scale_factor)
                
                u_val = U[orig_j, orig_i]
                v_val = V[orig_j, orig_i]
                magnitude = np.sqrt(u_val**2 + v_val**2)
                
                if magnitude < config.motion_threshold * avg_magnitude:
                    # Draw small dot for static regions
                    cv2.circle(visualization, (i, j), 0, (255, 0, 0), 1)
                else:
                    # Draw flow vector
                    scale = config.vector_scale * scale_factor * 10
                    
                    end_i = int(i + u_val * scale)
                    end_j = int(j + v_val * scale)
                    
                    end_i = max(0, min(scaled_w - 1, end_i))
                    end_j = max(0, min(scaled_h - 1, end_j))
                    
                    # Draw vector line
                    cv2.line(visualization, (i, j), (end_i, end_j), (0, 255, 0), 1)
                    
                    # Draw arrowhead
                    if end_i != i or end_j != j:
                        angle = np.arctan2(end_j - j, end_i - i)
                        arrow_head_size = 5
                        
                        arrow1_i = int(end_i - arrow_head_size * np.cos(angle + np.pi/6))
                        arrow1_j = int(end_j - arrow_head_size * np.sin(angle + np.pi/6))
                        arrow2_i = int(end_i - arrow_head_size * np.cos(angle - np.pi/6))
                        arrow2_j = int(end_j - arrow_head_size * np.sin(angle - np.pi/6))
                        
                        cv2.line(visualization, (end_i, end_j), (arrow1_i, arrow1_j), (0, 255, 0), 1)
                        cv2.line(visualization, (end_i, end_j), (arrow2_i, arrow2_j), (0, 255, 0), 1)
        
        return visualization
    
    @staticmethod
    def overlay_flow_on_difference(flow_bitmap: np.ndarray, diff_img: np.ndarray) -> np.ndarray:
        """
        Overlay flow vectors on difference image.
        
        Args:
            flow_bitmap: Flow visualization
            diff_img: Difference image
            
        Returns:
            Combined image
        """
        # Convert difference to RGB if grayscale
        if len(diff_img.shape) == 2:
            diff_rgb = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2RGB)
        else:
            diff_rgb = diff_img.copy()
        
        h, w = diff_img.shape[:2]
        flow_resized = cv2.resize(flow_bitmap, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create mask for flow vectors (non-white pixels)
        mask = np.any(flow_resized != [255, 255, 255], axis=2)
        
        # Overlay flow on difference
        result = diff_rgb.copy()
        result[mask] = flow_resized[mask]
        
        return result


class FlowProcessingWorker(QObject):
    """Worker for processing images with specified flow parameters."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    
    def __init__(self, base_image: np.ndarray, test_images: List[Tuple[str, np.ndarray]], 
                 config: FlowConfiguration, motion_regions: Optional[dict] = None):
        super().__init__()
        self.base_image = base_image
        self.test_images = test_images
        self.config = config
        self.abort_flag = False
        self.motion_regions = motion_regions or {}
        
        # Calculate regions if not provided
        if not self.motion_regions:
            from ..analysis.movement_analyzer import MotionAnalyzer
            analyzer = MotionAnalyzer()
            for filename, img in test_images:
                region_rect = analyzer.detect_moving_region(
                    base_image, img, 
                    self.config.block_detection_threshold, 
                    self.config.min_block_size
                )
                if region_rect:
                    self.motion_regions[filename] = region_rect
    
    def abort_processing(self):
        """Request abort of current processing."""
        self.abort_flag = True
    
    def process_images(self):
        """Process all images with current configuration."""
        if self.abort_flag:
            return
        
        results = []
        total = len(self.test_images)
        
        from ..core.image_operations import preprocess_image, compute_image_difference
        from ..core.flow_algorithms import basic_lucas_kanade_flow, hierarchical_lucas_kanade_flow
        
        for i, (filename, img) in enumerate(self.test_images):
            if self.abort_flag:
                return
            
            # Preprocess images
            preprocessed_base = preprocess_image(self.base_image, self.config)
            preprocessed_img = preprocess_image(img, self.config)
            
            # Determine algorithm
            use_hierarchical = 'R5' in filename or 'R10' in filename or 'R20' in filename or 'R40' in filename
            
            # Compute optical flow
            if use_hierarchical:
                U, V = hierarchical_lucas_kanade_flow(
                    preprocessed_base, 
                    preprocessed_img,
                    self.config.pyramid_levels,
                    int(self.config.window_size),
                    self.config.kernel_type,
                    self.config.gaussian_sigma,
                    self.config.interpolation_method,
                    self.config.border_mode
                )
            else:
                U, V = basic_lucas_kanade_flow(
                    preprocessed_base, 
                    preprocessed_img, 
                    int(self.config.window_size), 
                    self.config.kernel_type, 
                    self.config.gaussian_sigma
                )
            
            # Compute difference image
            diff = compute_image_difference(preprocessed_base, preprocessed_img)
            region_rect = self.motion_regions.get(filename)
            
            results.append((filename, img, preprocessed_img, diff, U, V, region_rect))
            
            self.progress.emit(int((i + 1) * 100 / total), f"Processing image {i+1}/{total}")
        
        self.finished.emit(results)
