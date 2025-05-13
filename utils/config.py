"""
Configuration management for the OpticalFlow application.
Contains parameter definitions and management logic.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List
import cv2


@dataclass
class FlowConfiguration:
    """Configuration for optical flow computation parameters."""
    
    # Preprocessing parameters
    gamma: float = 1.0
    gaussian_sigma: float = 1.0
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    gradient_scale: float = 1/8
    use_bilateral: bool = False
    temporal_kernel_size: int = 15
    
    # Lucas-Kanade algorithm parameters
    window_size: int = 15
    det_threshold: float = 1e-6
    magnitude_threshold: float = 0.1
    kernel_type: str = 'uniform'
    
    # Hierarchical flow parameters
    pyramid_levels: int = 3
    interpolation_method: int = cv2.INTER_CUBIC
    border_mode: int = cv2.BORDER_REFLECT101
    
    # Visualization parameters
    grid_spacing: int = 10
    vector_scale: float = 1.0
    motion_threshold: float = 0.1
    downsampling_factor: int = 20
    
    # Motion region detection
    block_detection_threshold: float = 0.1
    min_block_size: int = 20
    
    # Available options for UI dropdowns
    @staticmethod
    def get_kernel_types() -> List[Tuple[str, str]]:
        """Available kernel types for optical flow computation."""
        return [
            ('uniform', 'Uniform'),
            ('gaussian', 'Gaussian')
        ]
    
    @staticmethod
    def get_interpolation_methods() -> List[Tuple[int, str]]:
        """Available interpolation methods for warping."""
        return [
            (cv2.INTER_NEAREST, 'Nearest'),
            (cv2.INTER_LINEAR, 'Linear'),
            (cv2.INTER_CUBIC, 'Cubic'),
            (cv2.INTER_LANCZOS4, 'Lanczos4')
        ]
    
    @staticmethod
    def get_border_modes() -> List[Tuple[int, str]]:
        """Available border modes for image operations."""
        return [
            (cv2.BORDER_CONSTANT, 'Constant'),
            (cv2.BORDER_REPLICATE, 'Replicate'),
            (cv2.BORDER_REFLECT, 'Reflect'),
            (cv2.BORDER_WRAP, 'Wrap'),
            (cv2.BORDER_REFLECT101, 'Reflect_101')
        ]
    
    @classmethod
    def get_parameter_ranges(cls) -> Dict[str, Tuple[float, float]]:
        """Get valid ranges for all numeric parameters."""
        return {
            'gamma': (0.1, 3.0),
            'gaussian_sigma': (0.1, 5.0),
            'bilateral_d': (5, 15),
            'bilateral_sigma_color': (10.0, 150.0),
            'bilateral_sigma_space': (10.0, 150.0),
            'gradient_scale': (0.01, 1.0),
            'window_size': (5, 41),
            'temporal_kernel_size': (3, 31),
            'det_threshold': (1e-6, 1e-2),
            'magnitude_threshold': (0.1, 1.0),
            'grid_spacing': (5, 20),
            'vector_scale': (0.1, 10.0),
            'motion_threshold': (0.05, 0.5),
            'pyramid_levels': (1, 6),
            'downsampling_factor': (5, 100),
            'block_detection_threshold': (0.05, 0.5)
        }
    
    def update_parameter(self, param_name: str, value: Any):
        """Update a parameter value, ensuring it stays within valid range."""
        ranges = self.get_parameter_ranges()
        
        if param_name in ranges and isinstance(value, (int, float)):
            min_val, max_val = ranges[param_name]
            value = max(min_val, min(max_val, value))
            setattr(self, param_name, value)
        else:
            # For non-numeric parameters
            setattr(self, param_name, value)
    
    def to_api_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for API compatibility."""
        return {
            'k_size': self.window_size,
            'k_type': self.kernel_type,
            'sigma': self.gaussian_sigma,
            'levels': self.pyramid_levels,
            'interpolation': self.interpolation_method,
            'border_mode': self.border_mode
        }
    
    def from_dict(self, params_dict: Dict[str, Any]):
        """Load configuration from a dictionary."""
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def copy(self) -> 'FlowConfiguration':
        """Create a deep copy of the configuration."""
        copied = FlowConfiguration()
        
        for field_name, field_info in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            setattr(copied, field_name, value)
        
        return copied
