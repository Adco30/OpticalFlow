"""
Parameter optimization module for the OpticalFlow system.
Uses Bayesian optimization to find optimal parameters for optical flow computation.
"""
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import cv2
from skopt import gp_minimize
from skopt.space import Real, Integer
from typing import Dict, Tuple, Optional, List

from core.flow_algorithms import basic_lucas_kanade_flow, hierarchical_lucas_kanade_flow
from core.image_operations import preprocess_image, compute_image_difference
from analysis.movement_analyzer import MotionAnalyzer
from utils.config import FlowConfiguration


def compute_flow_quality_score(U: np.ndarray, V: np.ndarray, region_rect: Optional[Tuple[int, int, int, int]]) -> float:
    """
    Calculate quality score for optical flow based on:
    1. Minimal outliers outside detected region
    2. Consistent displacement vectors within region
    3. Uniform direction and magnitude within region
    
    Args:
        U, V: Flow components
        region_rect: Bounding box of detected motion region
        
    Returns:
        Quality score (higher is better)
    """
    if region_rect is None:
        return 0.0
        
    x, y, w, h = region_rect
    h_total, w_total = U.shape
    
    # Create masks for inside and outside the region
    inside_mask = np.zeros_like(U, dtype=bool)
    inside_mask[y:y+h, x:x+w] = True
    outside_mask = ~inside_mask
    
    # Calculate magnitudes
    magnitudes = np.sqrt(U**2 + V**2)
    
    # 1. Calculate outlier penalty
    inside_mean_mag = np.mean(magnitudes[inside_mask]) if np.any(inside_mask) else 0
    outside_mean_mag = np.mean(magnitudes[outside_mask]) if np.any(outside_mask) else 0
    
    outlier_ratio = outside_mean_mag / (inside_mean_mag + 1e-6) if inside_mean_mag > 0 else 1.0
    outlier_penalty = np.exp(-outlier_ratio * 5)
    
    # 2. Calculate coverage score
    inside_vectors = magnitudes[inside_mask]
    significant_vectors = inside_vectors > 0.1 * inside_mean_mag
    coverage_score = np.sum(significant_vectors) / (w * h)
    
    # 3. Calculate consistency
    direction_consistency = 0
    magnitude_consistency = 0
    
    if np.sum(inside_mask) > 0 and inside_mean_mag > 0:
        U_inside = U[inside_mask]
        V_inside = V[inside_mask]
        magnitudes_inside = magnitudes[inside_mask]
        
        significant_mask = magnitudes_inside > 0.1 * inside_mean_mag
        if np.sum(significant_mask) > 0:
            U_sig = U_inside[significant_mask]
            V_sig = V_inside[significant_mask]
            mag_sig = magnitudes_inside[significant_mask]
            
            # Direction consistency
            U_norm = U_sig / (mag_sig + 1e-6)
            V_norm = V_sig / (mag_sig + 1e-6)
            
            mean_U_dir = np.mean(U_norm)
            mean_V_dir = np.mean(V_norm)
            mean_dir_norm = np.sqrt(mean_U_dir**2 + mean_V_dir**2) + 1e-6
            mean_U_dir /= mean_dir_norm
            mean_V_dir /= mean_dir_norm
            
            dot_products = U_norm * mean_U_dir + V_norm * mean_V_dir
            direction_consistency = np.mean(dot_products)
            
            # Magnitude consistency
            mag_mean = np.mean(mag_sig)
            mag_std = np.std(mag_sig)
            magnitude_consistency = 1.0 / (1.0 + mag_std / mag_mean)
    
    # Calculate final score
    final_score = (
        0.4 * outlier_penalty +
        0.3 * coverage_score +
        0.2 * direction_consistency +
        0.1 * magnitude_consistency
    )
    
    return final_score


class ParameterOptimizer(QObject):
    """Worker class that handles parameter optimization using Bayesian optimization."""
    progress = pyqtSignal(int, str)
    optimized_params = pyqtSignal(object)
    finished = pyqtSignal(list)
    
    def __init__(self, base_image: np.ndarray, test_images: List[Tuple[str, np.ndarray]]):
        super().__init__()
        self.base_image = base_image
        self.test_images = test_images
        self.abort_flag = False
        
        # Initialize motion analyzer for region detection
        self.motion_analyzer = MotionAnalyzer()
        
        # Calculate and store motion regions for each image pair
        self.motion_regions = {}
        for filename, img in test_images:
            region_rect = self.motion_analyzer.detect_moving_region(base_image, img, 0.1, 20)
            if region_rect:
                self.motion_regions[filename] = region_rect
    
    def abort_optimization(self):
        """Request abort of current optimization."""
        self.abort_flag = True
    
    def run_optimization(self):
        """Execute Bayesian optimization for flow parameters."""
        if self.abort_flag:
            return
        
        # Define search space
        optimization_space = [
            Real(0.5, 2.5, name='gamma'),
            Real(0.1, 5.0, name='gaussian_sigma'),
            Integer(5, 41, name='window_size'),
            Real(0.05, 0.5, name='motion_threshold'),
            Real(0.5, 10.0, name='vector_scale'),
            Real(1/32, 1/2, name='gradient_scale'),
            Integer(1, 6, name='pyramid_levels'),
            Integer(3, 31, name='temporal_kernel_size'),
            Integer(5, 15, name='bilateral_d'),
            Real(10.0, 150.0, name='bilateral_sigma_color'),
            Real(10.0, 150.0, name='bilateral_sigma_space'),
            Real(1e-6, 1e-2, name='det_threshold')
        ]
        
        # Cache for parameter evaluations
        evaluation_cache = {}
        best_score = float('-inf')
        best_parameters = None
        
        self.progress.emit(0, "Starting parameter optimization...")
        
        def evaluate_parameters(params):
            """Evaluate a parameter set for optical flow quality."""
            if self.abort_flag:
                return float('-inf')
            
            # Unpack parameters
            gamma, gaussian_sigma, window_size, motion_threshold, vector_scale, gradient_scale, \
            pyramid_levels, temporal_kernel_size, bilateral_d, bilateral_sigma_color, \
            bilateral_sigma_space, det_threshold = params
            
            # Create cache key
            cache_key = (f"{gamma:.3f}_{gaussian_sigma:.3f}_{window_size}_{motion_threshold:.3f}_"
                        f"{vector_scale:.3f}_{gradient_scale:.6f}_{pyramid_levels}_{temporal_kernel_size}_"
                        f"{bilateral_d}_{bilateral_sigma_color:.1f}_{bilateral_sigma_space:.1f}_{det_threshold:.6f}")
            
            # Check cache
            if cache_key in evaluation_cache:
                return evaluation_cache[cache_key]
            
            # Create configuration object
            config = FlowConfiguration()
            config.gamma = gamma
            config.gaussian_sigma = gaussian_sigma
            config.window_size = window_size
            config.motion_threshold = motion_threshold
            config.vector_scale = vector_scale
            config.gradient_scale = gradient_scale
            config.use_bilateral = True
            config.temporal_kernel_size = temporal_kernel_size
            config.pyramid_levels = pyramid_levels
            config.kernel_type = 'gaussian'
            config.interpolation_method = cv2.INTER_CUBIC
            config.border_mode = cv2.BORDER_REFLECT101
            config.bilateral_d = bilateral_d
            config.bilateral_sigma_color = bilateral_sigma_color
            config.bilateral_sigma_space = bilateral_sigma_space
            config.det_threshold = det_threshold
            
            # Evaluate on all test images
            scores = []
            total_images = len(self.test_images)
            
            for i, (filename, img) in enumerate(self.test_images):
                if self.abort_flag:
                    return float('-inf')
                
                # Get motion region for this pair
                region_rect = self.motion_regions.get(filename)
                if not region_rect:
                    continue
                
                # Preprocess images
                preprocessed_base = preprocess_image(self.base_image.copy(), config)
                preprocessed_img = preprocess_image(img.copy(), config)
                
                # Determine algorithm based on filename
                use_hierarchical = 'R5' in filename or 'R10' in filename or 'R20' in filename or 'R40' in filename
                
                # Compute optical flow
                try:
                    if use_hierarchical:
                        U, V = hierarchical_lucas_kanade_flow(
                            preprocessed_base, 
                            preprocessed_img,
                            config.pyramid_levels,
                            int(config.window_size),
                            config.kernel_type,
                            config.gaussian_sigma,
                            config.interpolation_method,
                            config.border_mode
                        )
                    else:
                        U, V = basic_lucas_kanade_flow(
                            preprocessed_base, 
                            preprocessed_img, 
                            config.window_size, 
                            config.kernel_type, 
                            config.gaussian_sigma
                        )
                except Exception as e:
                    print(f"Error computing flow: {str(e)}")
                    return float('-inf')
                
                # Calculate quality score
                score = compute_flow_quality_score(U, V, region_rect)
                scores.append(score)
                
                # Update progress
                self.progress.emit(
                    int(i/total_images * 100),
                    f"Evaluating parameters {i+1}/{total_images}"
                )
            
            # Calculate average score
            avg_score = np.mean(scores) if scores else 0
            
            # Cache result
            evaluation_cache[cache_key] = avg_score
            
            # Track best parameters
            nonlocal best_score, best_parameters
            if avg_score > best_score:
                best_score = avg_score
                best_parameters = params
            
            return avg_score
        
        # Run Bayesian optimization
        try:
            result = gp_minimize(
                lambda x: -evaluate_parameters(x),  # Negate for minimization
                optimization_space,
                n_calls=50,
                n_random_starts=15,
                verbose=False,
                n_jobs=-1
            )
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            self.progress.emit(100, f"Optimization error: {str(e)}")
            return
        
        if self.abort_flag:
            return
        
        # Create optimized configuration
        optimized_config = self.create_optimized_config(result.x)
        
        # Emit optimized parameters
        self.optimized_params.emit(optimized_config)
        
        # Process images with optimized parameters
        self.process_with_optimized_params(optimized_config)
    
    def create_optimized_config(self, optimization_result: list) -> FlowConfiguration:
        """Create configuration object from optimization results."""
        config = FlowConfiguration()
        config.gamma = optimization_result[0]
        config.gaussian_sigma = optimization_result[1]
        config.window_size = optimization_result[2]
        config.motion_threshold = optimization_result[3]
        config.vector_scale = optimization_result[4]
        config.gradient_scale = optimization_result[5]
        config.pyramid_levels = optimization_result[6]
        config.temporal_kernel_size = optimization_result[7]
        config.bilateral_d = optimization_result[8]
        config.bilateral_sigma_color = optimization_result[9]
        config.bilateral_sigma_space = optimization_result[10]
        config.det_threshold = optimization_result[11]
        config.use_bilateral = True
        config.kernel_type = 'gaussian'
        
        return config
    
    def process_with_optimized_params(self, config: FlowConfiguration):
        """Process all images with optimized parameters."""
        if self.abort_flag:
            return
            
        results = []
        total = len(self.test_images)
        
        self.progress.emit(0, "Applying optimized parameters...")
        
        for i, (filename, img) in enumerate(self.test_images):
            if self.abort_flag:
                return
                
            # Preprocess images
            preprocessed_base = preprocess_image(self.base_image, config)
            preprocessed_img = preprocess_image(img, config)
            
            # Determine algorithm
            use_hierarchical = 'R5' in filename or 'R10' in filename or 'R20' in filename or 'R40' in filename
            
            # Compute optical flow
            if use_hierarchical:
                U, V = hierarchical_lucas_kanade_flow(
                    preprocessed_base, 
                    preprocessed_img,
                    config.pyramid_levels,
                    int(config.window_size),
                    config.kernel_type,
                    config.gaussian_sigma,
                    config.interpolation_method,
                    config.border_mode
                )
            else:
                U, V = basic_lucas_kanade_flow(
                    preprocessed_base, 
                    preprocessed_img, 
                    int(config.window_size), 
                    config.kernel_type, 
                    config.gaussian_sigma
                )
            
            # Compute difference image
            diff = compute_image_difference(preprocessed_base, preprocessed_img)
            
            # Get motion region for visualization
            region_rect = self.motion_regions.get(filename)
            
            results.append((filename, img, preprocessed_img, diff, U, V, region_rect))
            
            self.progress.emit(int((i + 1) * 100 / total), f"Processing image {i+1}/{total}")
            
        self.finished.emit(results)
