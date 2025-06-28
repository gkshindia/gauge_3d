"""
Gaussian Initialization for 4D Gaussian Splatting

This module handles the initialization of 3D Gaussians from point clouds
and RGB-D data for 4D Gaussian Splatting.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

try:
    import open3d as o3d
except ImportError:
    o3d = None

class GaussianInitializer:
    """Initialize 3D Gaussians from point cloud data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        # Gaussian parameters
        self.initial_opacity = self.config.get("initial_opacity", 0.1)
        self.initial_scale = self.config.get("initial_scale", 0.01)
        self.max_gaussians = self.config.get("max_gaussians", 1000000)
        
    def estimate_point_normals(self, points: np.ndarray) -> np.ndarray:
        """Estimate normals for point cloud using PCA"""
        if o3d is None:
            # Fallback: estimate normals using simple method
            return np.tile([0, 0, 1], (len(points), 1)).astype(np.float32)
        
        # Use Open3D for robust normal estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        return np.asarray(pcd.normals).astype(np.float32)
    
    def compute_scale_from_neighbors(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        """Compute initial scale for each Gaussian based on nearest neighbors"""
        if o3d is None:
            # Fallback: use fixed scale
            return np.full((len(points),), self.initial_scale, dtype=np.float32)
        
        # Build KD-tree for neighbor search
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        scales = []
        for i in range(len(points)):
            # Find k nearest neighbors
            [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k + 1)
            
            if len(idx) > 1:
                # Compute average distance to neighbors
                neighbor_points = points[idx[1:]]  # Exclude self
                distances = np.linalg.norm(neighbor_points - points[i], axis=1)
                avg_distance = np.mean(distances)
                scales.append(avg_distance / 3.0)  # Scale factor
            else:
                scales.append(self.initial_scale)
        
        return np.array(scales, dtype=np.float32)
    
    def initialize_gaussians_from_pointcloud(self, 
                                           point_cloud: np.ndarray,
                                           max_points: int = None) -> Dict[str, torch.Tensor]:
        """Initialize Gaussians from a single point cloud"""
        self.logger.info(f"Initializing Gaussians from point cloud with {len(point_cloud)} points")
        
        if len(point_cloud) == 0:
            raise ValueError("Empty point cloud provided")
        
        # Extract positions and colors
        positions = point_cloud[:, :3].astype(np.float32)
        colors = point_cloud[:, 3:6].astype(np.float32)
        
        # Subsample if too many points
        if max_points and len(positions) > max_points:
            indices = np.random.choice(len(positions), max_points, replace=False)
            positions = positions[indices]
            colors = colors[indices]
        
        num_points = len(positions)
        self.logger.info(f"Using {num_points} points for Gaussian initialization")
        
        # Estimate normals
        normals = self.estimate_point_normals(positions)
        
        # Compute scales based on local density
        scales = self.compute_scale_from_neighbors(positions)
        
        # Initialize rotations (quaternions) - start with identity
        rotations = np.tile([1, 0, 0, 0], (num_points, 1)).astype(np.float32)  # w, x, y, z
        
        # Initialize opacity (logit space for numerical stability)
        opacity_logits = np.full((num_points, 1), 
                                np.log(self.initial_opacity / (1 - self.initial_opacity)), 
                                dtype=np.float32)
        
        # Convert to torch tensors
        gaussian_params = {
            "positions": torch.from_numpy(positions).to(self.device),
            "colors": torch.from_numpy(colors).to(self.device),
            "scales": torch.from_numpy(scales.reshape(-1, 1).repeat(3, axis=1)).to(self.device),  # xyz scales
            "rotations": torch.from_numpy(rotations).to(self.device),  # quaternions
            "opacity": torch.from_numpy(opacity_logits).to(self.device),
            "normals": torch.from_numpy(normals).to(self.device)
        }
        
        self.logger.info(f"Initialized {num_points} Gaussians")
        return gaussian_params
    
    def initialize_temporal_gaussians(self, 
                                    point_clouds: List[np.ndarray],
                                    temporal_window: int = 10) -> List[Dict[str, torch.Tensor]]:
        """Initialize Gaussians for multiple frames with temporal consistency"""
        self.logger.info(f"Initializing temporal Gaussians for {len(point_clouds)} frames")
        
        if not point_clouds:
            raise ValueError("No point clouds provided")
        
        gaussians_sequence = []
        
        # Process frames in temporal windows
        for start_idx in range(0, len(point_clouds), temporal_window):
            end_idx = min(start_idx + temporal_window, len(point_clouds))
            window_clouds = point_clouds[start_idx:end_idx]
            
            self.logger.info(f"Processing temporal window {start_idx//temporal_window + 1}: "
                           f"frames {start_idx} to {end_idx-1}")
            
            # Initialize Gaussians for each frame in the window
            window_gaussians = []
            for i, point_cloud in enumerate(window_clouds):
                frame_idx = start_idx + i
                self.logger.info(f"Initializing frame {frame_idx}")
                
                try:
                    gaussian_params = self.initialize_gaussians_from_pointcloud(
                        point_cloud, 
                        max_points=self.max_gaussians // len(window_clouds)
                    )
                    
                    # Add temporal information
                    gaussian_params["frame_index"] = torch.tensor(frame_idx).to(self.device)
                    gaussian_params["temporal_window"] = torch.tensor(start_idx//temporal_window).to(self.device)
                    
                    window_gaussians.append(gaussian_params)
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize Gaussians for frame {frame_idx}: {e}")
                    continue
            
            gaussians_sequence.extend(window_gaussians)
        
        self.logger.info(f"Initialized Gaussians for {len(gaussians_sequence)} frames")
        return gaussians_sequence
    
    def create_temporal_correspondences(self, 
                                      gaussians_sequence: List[Dict[str, torch.Tensor]],
                                      correspondence_threshold: float = 0.05) -> Dict[str, Any]:
        """Create correspondences between Gaussians across frames"""
        self.logger.info("Creating temporal correspondences between Gaussians")
        
        correspondences = []
        
        for i in range(len(gaussians_sequence) - 1):
            current_gaussians = gaussians_sequence[i]
            next_gaussians = gaussians_sequence[i + 1]
            
            current_positions = current_gaussians["positions"].cpu().numpy()
            next_positions = next_gaussians["positions"].cpu().numpy()
            
            # Simple nearest neighbor correspondence
            frame_correspondences = []
            
            for j, pos in enumerate(current_positions):
                # Find nearest neighbor in next frame
                distances = np.linalg.norm(next_positions - pos, axis=1)
                nearest_idx = np.argmin(distances)
                
                if distances[nearest_idx] < correspondence_threshold:
                    frame_correspondences.append({
                        "frame1_idx": j,
                        "frame2_idx": nearest_idx,
                        "distance": float(distances[nearest_idx])
                    })
            
            correspondences.append({
                "frame1": i,
                "frame2": i + 1,
                "correspondences": frame_correspondences,
                "num_correspondences": len(frame_correspondences)
            })
            
            self.logger.info(f"Found {len(frame_correspondences)} correspondences "
                           f"between frames {i} and {i+1}")
        
        return {
            "temporal_correspondences": correspondences,
            "total_frames": len(gaussians_sequence)
        }
    
    def save_gaussians(self, 
                      gaussians: Dict[str, torch.Tensor], 
                      output_path: str,
                      metadata: Dict[str, Any] = None):
        """Save Gaussian parameters to file"""
        self.logger.info(f"Saving Gaussians to {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to numpy for saving
        gaussian_data = {}
        for key, tensor in gaussians.items():
            if isinstance(tensor, torch.Tensor):
                gaussian_data[key] = tensor.cpu().numpy()
            else:
                gaussian_data[key] = tensor
        
        # Add metadata
        if metadata:
            gaussian_data["metadata"] = metadata
        
        # Save as compressed numpy file
        np.savez_compressed(output_path, **gaussian_data)
        
        self.logger.info(f"Saved Gaussians with {len(gaussians['positions'])} points")
    
    def load_gaussians(self, input_path: str) -> Dict[str, torch.Tensor]:
        """Load Gaussian parameters from file"""
        self.logger.info(f"Loading Gaussians from {input_path}")
        
        data = np.load(input_path)
        
        gaussians = {}
        for key in data.files:
            if key != "metadata":
                gaussians[key] = torch.from_numpy(data[key]).to(self.device)
        
        self.logger.info(f"Loaded Gaussians with {len(gaussians['positions'])} points")
        return gaussians

def main():
    """Main entry point for Gaussian initialization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize Gaussians for 4D Gaussian Splatting")
    parser.add_argument("video_name", help="Name of video to process")
    parser.add_argument("--data-dir", default="output/gaussian_data", help="Data directory")
    parser.add_argument("--output-dir", default="output/gaussian_init", help="Output directory")
    parser.add_argument("--max-gaussians", type=int, default=1000000, help="Maximum number of Gaussians")
    parser.add_argument("--temporal-window", type=int, default=10, help="Temporal window size")
    
    args = parser.parse_args()
    
    config = {
        "max_gaussians": args.max_gaussians,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    initializer = GaussianInitializer(config)
    
    try:
        # Load point clouds
        data_dir = Path(args.data_dir)
        point_clouds_dir = data_dir / "point_clouds" / args.video_name
        
        if not point_clouds_dir.exists():
            raise ValueError(f"Point clouds directory not found: {point_clouds_dir}")
        
        # Load point cloud files
        pc_files = sorted(list(point_clouds_dir.glob("*.npy")))
        point_clouds = []
        
        for pc_file in pc_files:
            pc = np.load(pc_file)
            point_clouds.append(pc)
        
        print(f"Loaded {len(point_clouds)} point clouds")
        
        # Initialize Gaussians
        gaussians_sequence = initializer.initialize_temporal_gaussians(
            point_clouds, args.temporal_window
        )
        
        # Create correspondences
        correspondences = initializer.create_temporal_correspondences(gaussians_sequence)
        
        # Save results
        output_dir = Path(args.output_dir) / args.video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Gaussians for each frame
        for i, gaussians in enumerate(gaussians_sequence):
            output_file = output_dir / f"frame_{i:06d}_gaussians.npz"
            initializer.save_gaussians(gaussians, str(output_file))
        
        # Save correspondences
        correspondences_file = output_dir / "temporal_correspondences.json"
        with open(correspondences_file, 'w') as f:
            json.dump(correspondences, f, indent=2)
        
        print(f"✅ Successfully initialized Gaussians for {len(gaussians_sequence)} frames")
        
    except Exception as e:
        print(f"❌ Failed to initialize Gaussians: {e}")
        raise

if __name__ == "__main__":
    main()
