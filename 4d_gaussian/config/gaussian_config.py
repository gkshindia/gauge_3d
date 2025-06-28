"""
4D Gaussian Splatting Configuration

This module contains configuration settings for the 4D Gaussian Splatting pipeline.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path

@dataclass
class GaussianConfig:
    """Configuration for 4D Gaussian Splatting pipeline"""
    
    # Model Configuration
    model_name: str = "4d-gaussians-base"
    device: str = "cuda"  # cuda, cpu, mps
    precision: str = "fp16"  # fp32, fp16, bf16
    
    # Gaussian Parameters
    initial_opacity: float = 0.1
    initial_scale: float = 0.01
    max_gaussians: int = 1000000
    min_gaussians: int = 10000
    
    # Temporal Parameters
    temporal_window: int = 10  # Number of frames to process together
    temporal_consistency_weight: float = 0.1
    motion_threshold: float = 0.02
    
    # Optimization Parameters
    learning_rate: float = 0.01
    num_iterations: int = 1000
    batch_size: int = 4
    gradient_clip: float = 1.0
    
    # Rendering Parameters
    render_resolution: Tuple[int, int] = (1920, 1080)
    render_fov: float = 60.0
    near_plane: float = 0.1
    far_plane: float = 100.0
    
    # Quality Settings
    depth_threshold: float = 0.01
    color_threshold: float = 0.05
    normal_threshold: float = 0.1
    
    # I/O Settings
    output_format: str = "ply"  # ply, obj, gaussian
    save_intermediate: bool = True
    compression_level: int = 6
    
    # Camera Settings
    estimate_poses: bool = True
    use_colmap: bool = True
    intrinsic_matrix: Optional[List[List[float]]] = None
    
    # Performance Settings
    use_mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

@dataclass 
class DataConfig:
    """Configuration for data preparation and loading"""
    
    # Input paths
    frames_dir: str = "output/frames"
    depth_maps_dir: str = "output/depth_maps"
    depth_stats_file: str = "output/depth_stats"
    
    # Output paths
    gaussian_output_dir: str = "output/gaussian_reconstruction"
    point_cloud_dir: str = "output/point_clouds"
    camera_params_dir: str = "output/camera_params"
    
    # Processing settings
    frame_skip: int = 1  # Process every nth frame
    max_frames: Optional[int] = None
    crop_borders: int = 0  # Pixels to crop from borders
    
    # Quality filters
    min_depth: float = 0.1
    max_depth: float = 50.0
    depth_percentile_filter: float = 0.99  # Remove outliers
    
    # Preprocessing
    apply_bilateral_filter: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75

def get_default_config() -> GaussianConfig:
    """Get default configuration for 4D Gaussian Splatting"""
    return GaussianConfig()

def get_data_config() -> DataConfig:
    """Get default data configuration"""
    return DataConfig()

def load_config_from_file(config_path: str) -> GaussianConfig:
    """Load configuration from YAML file"""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return GaussianConfig(**config_dict)

def save_config_to_file(config: GaussianConfig, config_path: str):
    """Save configuration to YAML file"""
    import yaml
    from dataclasses import asdict
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False, indent=2)
