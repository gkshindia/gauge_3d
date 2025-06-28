"""
Transformer Enhancement Module for Gauge 3D

This module implements Phase 3 of the Gauge 3D pipeline: Transformer Enhancement 
with P4Transformer for advanced point cloud processing and reconstruction.

Components:
- Point Cloud Extraction: Extract point clouds from 4D Gaussians
- P4Transformer Integration: Use P4Transformer for point cloud enhancement
- Enhancement Pipeline: Denoising, completion, and feature enhancement
- Reconstruction Pipeline: Convert enhanced point clouds back to Gaussians

The transformer enhancement improves the quality and consistency of the 
4D Gaussian reconstruction through advanced neural processing.
"""

from .point_cloud_extractor import PointCloudExtractor
from .p4transformer_integration import P4TransformerIntegration
from .enhancement_pipeline import EnhancementPipeline
from .reconstruction_pipeline import ReconstructionPipeline

__version__ = "1.0.0"
__author__ = "Gauge 3D Team"

__all__ = [
    "PointCloudExtractor",
    "P4TransformerIntegration", 
    "EnhancementPipeline",
    "ReconstructionPipeline"
]
