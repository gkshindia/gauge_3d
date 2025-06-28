"""
4D Gaussian Splatting Pipeline for Video-to-3D Reconstruction

This module implements 4D Gaussian Splatting for dynamic scene reconstruction
from RGB-D video sequences.
"""

__version__ = "0.1.0"
__author__ = "Gauge 3D Team"

from .setup.environment_setup import GaussianEnvironmentSetup
from .data_preparation.data_converter import DepthToGaussianConverter
from .gaussian_generation.gaussian_initializer import GaussianInitializer
from .optimization.temporal_optimizer import TemporalGaussianOptimizer
from .rendering.gaussian_renderer import GaussianRenderer
from .utils.visualization import GaussianVisualizer

__all__ = [
    "GaussianEnvironmentSetup",
    "DepthToGaussianConverter", 
    "GaussianInitializer",
    "TemporalGaussianOptimizer",
    "GaussianRenderer",
    "GaussianVisualizer"
]
