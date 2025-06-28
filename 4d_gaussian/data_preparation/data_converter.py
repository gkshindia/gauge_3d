"""
Data Preparation for 4D Gaussian Splatting

This module handles the conversion of RGB frames and depth maps to the format
required for 4D Gaussian Splatting, including camera pose estimation and
point cloud generation.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

try:
    import open3d as o3d
except ImportError:
    o3d = None

class DepthToGaussianConverter:
    """Convert depth maps and RGB frames to Gaussian Splatting format"""
    
    def __init__(self, 
                 frames_dir: str,
                 depth_maps_dir: str,
                 output_dir: str,
                 config: Dict[str, Any] = None):
        self.frames_dir = Path(frames_dir)
        self.depth_maps_dir = Path(depth_maps_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Point cloud optimization settings
        self.quality_presets = {
            "preview": {
                "max_points_per_frame": 10000,
                "depth_downsample": 4,
                "filter_aggressive": True,
                "description": "Fast preview mode"
            },
            "standard": {
                "max_points_per_frame": 200000,
                "depth_downsample": 2,
                "filter_aggressive": False,
                "description": "Balanced quality and performance"
            },
            "high": {
                "max_points_per_frame": 500000,
                "depth_downsample": 1,
                "filter_aggressive": False,
                "description": "High quality reconstruction"
            },
            "ultra": {
                "max_points_per_frame": 1000000,
                "depth_downsample": 1,
                "filter_aggressive": False,
                "description": "Maximum quality (slow)"
            }
        }
        
        # Set quality preset
        quality = self.config.get("quality", "standard")
        if quality in self.quality_presets:
            self.quality_config = self.quality_presets[quality]
            self.logger.info(f"Using quality preset '{quality}': {self.quality_config['description']}")
        else:
            self.quality_config = self.quality_presets["standard"]
            self.logger.warning(f"Unknown quality preset '{quality}', using 'standard'")
        
        # Default camera parameters
        self.default_intrinsics = {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 960.0,
            "cy": 540.0,
            "width": 1920,
            "height": 1080
        }
        
    def load_frame_and_depth(self, frame_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load RGB frame and corresponding depth map"""
        # Load RGB frame
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Could not load frame: {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Load depth map
        depth = np.load(depth_path)
        if depth is None:
            raise ValueError(f"Could not load depth map: {depth_path}")
        
        # Ensure depth is in meters and proper range
        if depth.max() <= 1.0:
            # Depth is normalized, scale to meters
            depth = depth * 50.0  # Assume max depth of 50m
        
        return frame, depth
    
    def estimate_camera_poses_simple(self, num_frames: int) -> List[np.ndarray]:
        """Simple camera pose estimation - circular trajectory"""
        poses = []
        
        # Create circular camera trajectory
        radius = 3.0
        height = 1.5
        
        for i in range(num_frames):
            angle = 2 * np.pi * i / max(num_frames - 1, 1)
            
            # Camera position
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = height
            
            # Look at center
            camera_pos = np.array([x, y, z])
            target_pos = np.array([0, 0, 0])
            up_vector = np.array([0, 1, 0])
            
            # Compute rotation matrix
            forward = target_pos - camera_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, up_vector)
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            
            rotation = np.column_stack([right, up, -forward])
            
            # Create 4x4 transformation matrix
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = camera_pos
            
            poses.append(pose)
        
        return poses
    
    def run_colmap(self, images_dir: str, output_dir: str) -> bool:
        """Run COLMAP for camera pose estimation"""
        self.logger.info("Running COLMAP for camera pose estimation...")
        
        try:
            import subprocess
            
            colmap_output = Path(output_dir) / "colmap"
            colmap_output.mkdir(parents=True, exist_ok=True)
            
            # Feature extraction
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", str(colmap_output / "database.db"),
                "--image_path", str(images_dir),
                "--ImageReader.single_camera", "1"
            ], check=True)
            
            # Feature matching
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(colmap_output / "database.db")
            ], check=True)
            
            # Sparse reconstruction
            sparse_dir = colmap_output / "sparse"
            sparse_dir.mkdir(exist_ok=True)
            
            subprocess.run([
                "colmap", "mapper",
                "--database_path", str(colmap_output / "database.db"),
                "--image_path", str(images_dir),
                "--output_path", str(sparse_dir)
            ], check=True)
            
            self.logger.info("COLMAP reconstruction completed successfully")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.warning(f"COLMAP failed: {e}")
            return False
    
    def depth_to_point_cloud(self, 
                            rgb_image: np.ndarray,
                            depth_map: np.ndarray,
                            intrinsics: Dict[str, float],
                            pose: np.ndarray = None) -> np.ndarray:
        """Convert depth map to colored point cloud with density optimization"""
        
        # Apply downsampling if specified
        downsample = self.quality_config.get("depth_downsample", 1)
        if downsample > 1:
            rgb_image = rgb_image[::downsample, ::downsample]
            depth_map = depth_map[::downsample, ::downsample]
            # Adjust intrinsics for downsampling
            intrinsics = intrinsics.copy()
            intrinsics["fx"] /= downsample
            intrinsics["fy"] /= downsample
            intrinsics["cx"] /= downsample
            intrinsics["cy"] /= downsample
        
        h, w = depth_map.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert to camera coordinates
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        
        # Remove invalid depth values
        valid_mask = (depth_map > 0.1) & (depth_map < 50.0)
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_map[valid_mask]
        
        # Convert to 3D coordinates
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid
        
        # Stack coordinates
        points_3d = np.stack([x, y, z], axis=1)
        
        # Get colors
        colors = rgb_image[valid_mask] / 255.0
        
        # Apply camera pose transformation if provided
        if pose is not None:
            # Convert to homogeneous coordinates
            points_homogeneous = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1))], axis=1)
            # Transform points
            points_transformed = (pose @ points_homogeneous.T).T
            points_3d = points_transformed[:, :3]
        
        # Combine points and colors
        point_cloud = np.concatenate([points_3d, colors], axis=1)
        
        # Apply max point limit
        max_points = self.quality_config.get("max_points_per_frame", 200000)
        if len(point_cloud) > max_points:
            # Randomly sample points to meet limit
            indices = np.random.choice(len(point_cloud), max_points, replace=False)
            point_cloud = point_cloud[indices]
            self.logger.info(f"Downsampled point cloud from {len(points_3d)} to {max_points} points")
        
        return point_cloud
    
    def filter_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Filter point cloud to remove outliers and noise"""
        if o3d is None:
            self.logger.warning("Open3D not available, skipping point cloud filtering")
            return point_cloud
        
        # Convert to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
        
        # Apply filtering based on quality settings
        aggressive_filter = self.quality_config.get("filter_aggressive", False)
        
        if aggressive_filter:
            # More aggressive filtering for preview mode
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
            pcd, _ = pcd.remove_radius_outlier(nb_points=8, radius=0.1)
        else:
            # Standard filtering
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        
        # Convert back to numpy
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        filtered_cloud = np.concatenate([points, colors], axis=1)
        reduction = len(point_cloud) - len(filtered_cloud)
        if reduction > 0:
            self.logger.info(f"Filtered out {reduction} outlier points ({reduction/len(point_cloud)*100:.1f}%)")
        
        return filtered_cloud
    
    def process_video_sequence(self, video_name: str) -> Dict[str, Any]:
        """Process complete video sequence for Gaussian Splatting"""
        self.logger.info(f"Processing video sequence: {video_name}")
        
        # Find frame and depth directories
        frame_dir = self.frames_dir / video_name
        depth_dir = self.depth_maps_dir / video_name
        
        if not frame_dir.exists() or not depth_dir.exists():
            raise ValueError(f"Frame or depth directory not found for {video_name}")
        
        # Get frame and depth file lists
        frame_files = sorted(list(frame_dir.glob("*.jpg")))
        depth_files = sorted(list(depth_dir.glob("*_depth.npy")))
        
        # Create frame-depth pairs by matching frame numbers
        frame_depth_pairs = []
        for depth_file in depth_files:
            # Extract frame number from depth filename: frame_000123_depth.npy -> 000123
            depth_name = depth_file.name
            if depth_name.startswith("frame_") and depth_name.endswith("_depth.npy"):
                frame_num = depth_name[6:-10]  # Extract the number part
                corresponding_frame = frame_dir / f"frame_{frame_num}.jpg"
                
                if corresponding_frame.exists():
                    frame_depth_pairs.append((corresponding_frame, depth_file))
                else:
                    self.logger.warning(f"No corresponding frame found for {depth_file.name}")
        
        self.logger.info(f"Found {len(frame_depth_pairs)} valid frame-depth pairs")
        num_frames = len(frame_depth_pairs)
        
        # Estimate camera poses
        try_colmap = self.config.get("use_colmap", True)
        if try_colmap and self.run_colmap(str(frame_dir), str(self.output_dir)):
            # TODO: Load COLMAP poses
            self.logger.info("Using COLMAP poses (implementation needed)")
            poses = self.estimate_camera_poses_simple(num_frames)
        else:
            self.logger.info("Using simple circular trajectory")
            poses = self.estimate_camera_poses_simple(num_frames)
        
        # Process each frame
        point_clouds = []
        frame_data = []
        
        for i, (frame_file, depth_file) in enumerate(frame_depth_pairs):
            pose = poses[i] if i < len(poses) else poses[-1]
            
            self.logger.info(f"Processing frame {i+1}/{num_frames}: {frame_file.name} + {depth_file.name}")
            
            try:
                # Load frame and depth
                rgb_image, depth_map = self.load_frame_and_depth(str(frame_file), str(depth_file))
                
                # Convert to point cloud
                point_cloud = self.depth_to_point_cloud(
                    rgb_image, depth_map, self.default_intrinsics, pose
                )
                
                # Filter point cloud
                if self.config.get("filter_point_clouds", True):
                    point_cloud = self.filter_point_cloud(point_cloud)
                
                point_clouds.append(point_cloud)
                
                # Store frame metadata
                frame_data.append({
                    "frame_index": i,
                    "frame_path": str(frame_file),
                    "depth_path": str(depth_file),
                    "pose": pose.tolist(),
                    "intrinsics": self.default_intrinsics,
                    "point_count": len(point_cloud)
                })
                
            except Exception as e:
                self.logger.error(f"Failed to process frame {i}: {e}")
                continue
        
        # Save results
        output_data = {
            "video_name": video_name,
            "num_frames": len(frame_data),
            "frames": frame_data,
            "camera_intrinsics": self.default_intrinsics
        }
        
        # Save point clouds
        point_clouds_dir = self.output_dir / "point_clouds" / video_name
        point_clouds_dir.mkdir(parents=True, exist_ok=True)
        
        for i, pc in enumerate(point_clouds):
            pc_file = point_clouds_dir / f"frame_{i:06d}_pointcloud.npy"
            np.save(pc_file, pc)
        
        # Save metadata
        metadata_file = self.output_dir / f"{video_name}_gaussian_data.json"
        with open(metadata_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Processed {len(frame_data)} frames for {video_name}")
        return output_data

def main():
    """Main entry point for data preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for 4D Gaussian Splatting")
    parser.add_argument("video_name", help="Name of video to process")
    parser.add_argument("--frames-dir", default="output/frames", help="Frames directory")
    parser.add_argument("--depth-dir", default="output/depth_maps", help="Depth maps directory")
    parser.add_argument("--output-dir", default="output/gaussian_data", help="Output directory")
    parser.add_argument("--use-colmap", action="store_true", help="Use COLMAP for pose estimation")
    
    args = parser.parse_args()
    
    config = {
        "use_colmap": args.use_colmap,
        "filter_point_clouds": True
    }
    
    converter = DepthToGaussianConverter(
        args.frames_dir, args.depth_dir, args.output_dir, config
    )
    
    try:
        result = converter.process_video_sequence(args.video_name)
        print(f"✅ Successfully processed {result['num_frames']} frames")
    except Exception as e:
        print(f"❌ Failed to process video: {e}")
        raise

if __name__ == "__main__":
    main()
