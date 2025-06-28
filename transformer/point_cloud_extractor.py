#!/usr/bin/env python3
"""
Point Cloud Extractor for Transformer Enhancement

This module handles the extraction of high-quality point clouds from 4D Gaussians
for input into the P4Transformer enhancement pipeline.

Key Functions:
- Extract temporal point cloud sequences from 4D Gaussians
- Apply quality filtering and consistency checks
- Generate metadata for downstream processing
- Handle batch processing for large datasets
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

# Optional imports (will be handled gracefully if not available)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Point cloud extraction will use CPU-only mode.")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. Some point cloud operations will be limited.")

logger = logging.getLogger(__name__)


class PointCloudExtractor:
    """
    Extracts point clouds from 4D Gaussian data for transformer enhancement.
    
    This class handles the conversion of 4D Gaussian splats into structured
    point cloud representations suitable for P4Transformer processing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the point cloud extractor.
        
        Args:
            config: Configuration dictionary with extraction parameters
        """
        self.config = config or self._get_default_config()
        self.device = self._setup_device()
        
        logger.info(f"PointCloudExtractor initialized with device: {self.device}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for point cloud extraction"""
        return {
            "min_points_per_frame": 1000,
            "max_points_per_frame": 50000,
            "temporal_consistency_threshold": 0.8,
            "quality_threshold": 0.5,
            "voxel_size": 0.01,
            "outlier_removal": True,
            "batch_size": 4
        }
    
    def _setup_device(self) -> str:
        """Setup computation device (CUDA/CPU)"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def extract_from_gaussians(
        self, 
        gaussian_data: Union[str, Path, Dict],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Extract point clouds from 4D Gaussian data.
        
        Args:
            gaussian_data: Path to Gaussian data file or loaded data dict
            output_dir: Directory to save extracted point clouds
            
        Returns:
            Dictionary containing extracted point clouds and metadata
        """
        logger.info("Starting point cloud extraction from 4D Gaussians")
        
        # Load Gaussian data if path provided
        if isinstance(gaussian_data, (str, Path)):
            gaussian_data = self._load_gaussian_data(gaussian_data)
        
        # Extract point clouds frame by frame
        point_clouds = []
        metadata = {
            "num_frames": 0,
            "total_points": 0,
            "quality_scores": [],
            "temporal_consistency": 0.0
        }
        
        for frame_idx, frame_data in enumerate(gaussian_data.get("frames", [])):
            logger.debug(f"Extracting frame {frame_idx}")
            
            # Extract point cloud from Gaussian parameters
            point_cloud = self._extract_frame_points(frame_data)
            
            # Apply quality filtering
            filtered_cloud = self._filter_point_cloud(point_cloud)
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(filtered_cloud)
            
            if quality_score >= self.config["quality_threshold"]:
                point_clouds.append(filtered_cloud)
                metadata["quality_scores"].append(quality_score)
                metadata["total_points"] += len(filtered_cloud["points"])
                
                logger.debug(f"Frame {frame_idx}: {len(filtered_cloud['points'])} points, quality: {quality_score:.3f}")
            else:
                logger.warning(f"Frame {frame_idx} rejected due to low quality: {quality_score:.3f}")
        
        metadata["num_frames"] = len(point_clouds)
        
        # Calculate temporal consistency
        if len(point_clouds) > 1:
            metadata["temporal_consistency"] = self._calculate_temporal_consistency(point_clouds)
        
        # Save results if output directory specified
        if output_dir:
            self._save_point_clouds(point_clouds, metadata, output_dir)
        
        logger.info(f"Extracted {metadata['num_frames']} point clouds with {metadata['total_points']} total points")
        
        return {
            "point_clouds": point_clouds,
            "metadata": metadata,
            "config": self.config
        }
    
    def _load_gaussian_data(self, data_path: Union[str, Path]) -> Dict:
        """Load 4D Gaussian data from file"""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Gaussian data file not found: {data_path}")
        
        logger.info(f"Loading Gaussian data from: {data_path}")
        
        # Handle different file formats
        if data_path.suffix == ".pth":
            if TORCH_AVAILABLE:
                return torch.load(data_path, map_location="cpu")
            else:
                raise ImportError("PyTorch required to load .pth files")
        elif data_path.suffix == ".npy":
            return {"frames": [{"data": np.load(data_path)}]}
        elif data_path.suffix == ".json":
            import json
            with open(data_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def _extract_frame_points(self, frame_data: Dict) -> Dict:
        """Extract point cloud from a single frame's Gaussian data"""
        # Extract Gaussian parameters
        positions = frame_data.get("positions", frame_data.get("means", []))
        colors = frame_data.get("colors", [])
        opacities = frame_data.get("opacities", [])
        scales = frame_data.get("scales", [])
        
        # Convert to numpy arrays
        if TORCH_AVAILABLE and torch.is_tensor(positions):
            positions = positions.cpu().numpy()
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        
        # Create point cloud structure
        point_cloud = {
            "points": positions,
            "colors": np.array(colors) if len(colors) > 0 else np.ones((len(positions), 3)),
            "opacities": np.array(opacities) if len(opacities) > 0 else np.ones(len(positions)),
            "scales": np.array(scales) if len(scales) > 0 else np.ones((len(positions), 3))
        }
        
        return point_cloud
    
    def _filter_point_cloud(self, point_cloud: Dict) -> Dict:
        """Apply filtering to improve point cloud quality"""
        points = point_cloud["points"]
        
        # Remove outliers if enabled
        if self.config["outlier_removal"] and OPEN3D_AVAILABLE:
            # Convert to Open3D format
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Statistical outlier removal
            pcd, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Filter all arrays based on inliers
            filtered_cloud = {}
            for key, values in point_cloud.items():
                if isinstance(values, np.ndarray) and len(values) == len(points):
                    filtered_cloud[key] = values[inliers]
                else:
                    filtered_cloud[key] = values
        else:
            filtered_cloud = point_cloud.copy()
        
        # Subsample if too many points
        num_points = len(filtered_cloud["points"])
        max_points = self.config["max_points_per_frame"]
        
        if num_points > max_points:
            indices = np.random.choice(num_points, max_points, replace=False)
            for key, values in filtered_cloud.items():
                if isinstance(values, np.ndarray) and len(values) == num_points:
                    filtered_cloud[key] = values[indices]
        
        return filtered_cloud
    
    def _calculate_quality_score(self, point_cloud: Dict) -> float:
        """Calculate quality score for a point cloud"""
        points = point_cloud["points"]
        
        if len(points) < self.config["min_points_per_frame"]:
            return 0.0
        
        # Basic quality metrics
        score = 1.0
        
        # Point density score
        density_score = min(len(points) / self.config["max_points_per_frame"], 1.0)
        score *= density_score
        
        # Spatial distribution score (measure of coverage)
        if len(points) > 10:
            coords_std = np.std(points, axis=0)
            distribution_score = min(np.mean(coords_std) / 0.1, 1.0)  # Normalize by expected scale
            score *= distribution_score
        
        # Opacity-based quality (if available)
        if "opacities" in point_cloud:
            opacity_score = np.mean(point_cloud["opacities"])
            score *= opacity_score
        
        return min(score, 1.0)
    
    def _calculate_temporal_consistency(self, point_clouds: List[Dict]) -> float:
        """Calculate temporal consistency across frames"""
        if len(point_clouds) < 2:
            return 1.0
        
        consistency_scores = []
        
        for i in range(1, len(point_clouds)):
            prev_points = point_clouds[i-1]["points"]
            curr_points = point_clouds[i]["points"]
            
            # Simple consistency metric based on point count and spatial distribution
            count_ratio = min(len(curr_points), len(prev_points)) / max(len(curr_points), len(prev_points))
            
            # Spatial consistency (simplified)
            prev_center = np.mean(prev_points, axis=0)
            curr_center = np.mean(curr_points, axis=0)
            spatial_consistency = 1.0 / (1.0 + np.linalg.norm(curr_center - prev_center))
            
            frame_consistency = (count_ratio + spatial_consistency) / 2.0
            consistency_scores.append(frame_consistency)
        
        return np.mean(consistency_scores)
    
    def _save_point_clouds(
        self, 
        point_clouds: List[Dict], 
        metadata: Dict, 
        output_dir: Union[str, Path]
    ):
        """Save extracted point clouds to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(point_clouds)} point clouds to: {output_dir}")
        
        # Save individual point clouds
        for i, point_cloud in enumerate(point_clouds):
            # Save as NPY format
            frame_dir = output_dir / f"frame_{i:04d}"
            frame_dir.mkdir(exist_ok=True)
            
            np.save(frame_dir / "points.npy", point_cloud["points"])
            np.save(frame_dir / "colors.npy", point_cloud["colors"])
            
            if "opacities" in point_cloud:
                np.save(frame_dir / "opacities.npy", point_cloud["opacities"])
            if "scales" in point_cloud:
                np.save(frame_dir / "scales.npy", point_cloud["scales"])
            
            # Save as PLY if Open3D available
            if OPEN3D_AVAILABLE:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(point_cloud["points"])
                pcd.colors = o3d.utility.Vector3dVector(point_cloud["colors"])
                o3d.io.write_point_cloud(str(frame_dir / "point_cloud.ply"), pcd)
        
        # Save metadata
        import json
        with open(output_dir / "metadata.json", 'w') as f:
            # Convert numpy types to JSON serializable
            serializable_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, np.ndarray):
                    serializable_metadata[key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32)):
                    serializable_metadata[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    serializable_metadata[key] = float(value)
                else:
                    serializable_metadata[key] = value
            
            json.dump(serializable_metadata, f, indent=2)
        
        logger.info("Point cloud extraction complete")


def main():
    """Command line interface for point cloud extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract point clouds from 4D Gaussians")
    parser.add_argument("input", help="Path to 4D Gaussian data")
    parser.add_argument("--output", "-o", help="Output directory for point clouds")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Load configuration if provided
    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize extractor and extract point clouds
    extractor = PointCloudExtractor(config)
    result = extractor.extract_from_gaussians(args.input, args.output)
    
    print(f"✅ Extracted {result['metadata']['num_frames']} point clouds")
    print(f"   Total points: {result['metadata']['total_points']}")
    print(f"   Temporal consistency: {result['metadata']['temporal_consistency']:.3f}")


if __name__ == "__main__":
    main()
