"""
Visualization utilities for 4D Gaussian Splatting

This module provides visualization tools for Gaussian parameters,
point clouds, and rendering results.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import json

try:
    import open3d as o3d
except ImportError:
    o3d = None

class GaussianVisualizer:
    """Visualization tools for 4D Gaussian Splatting"""
    
    def __init__(self, output_dir: str = "output/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def visualize_point_cloud(self, 
                             point_cloud: np.ndarray, 
                             title: str = "Point Cloud",
                             save_path: str = None) -> None:
        """Visualize point cloud with colors"""
        if o3d is None:
            self.logger.warning("Open3D not available, cannot visualize point cloud")
            return
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        if point_cloud.shape[1] >= 6:
            pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])
        
        # Visualize
        if save_path:
            # Save as PLY file
            o3d.io.write_point_cloud(save_path, pcd)
            self.logger.info(f"Point cloud saved to {save_path}")
        else:
            # Interactive visualization
            o3d.visualization.draw_geometries([pcd], window_name=title)
    
    def plot_gaussian_statistics(self, 
                                gaussians: Dict[str, np.ndarray],
                                save_path: str = None) -> None:
        """Plot statistics of Gaussian parameters"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Gaussian Parameters Statistics", fontsize=16)
        
        # Positions distribution
        positions = gaussians["positions"]
        axes[0, 0].hist(positions[:, 0], bins=50, alpha=0.7, label="X")
        axes[0, 0].hist(positions[:, 1], bins=50, alpha=0.7, label="Y")
        axes[0, 0].hist(positions[:, 2], bins=50, alpha=0.7, label="Z")
        axes[0, 0].set_title("Position Distribution")
        axes[0, 0].legend()
        
        # Colors distribution
        if "colors" in gaussians:
            colors = gaussians["colors"]
            axes[0, 1].hist(colors[:, 0], bins=50, alpha=0.7, color="red", label="R")
            axes[0, 1].hist(colors[:, 1], bins=50, alpha=0.7, color="green", label="G")
            axes[0, 1].hist(colors[:, 2], bins=50, alpha=0.7, color="blue", label="B")
            axes[0, 1].set_title("Color Distribution")
            axes[0, 1].legend()
        
        # Scales distribution
        if "scales" in gaussians:
            scales = gaussians["scales"]
            if scales.shape[1] >= 3:
                axes[0, 2].hist(scales[:, 0], bins=50, alpha=0.7, label="Scale X")
                axes[0, 2].hist(scales[:, 1], bins=50, alpha=0.7, label="Scale Y")
                axes[0, 2].hist(scales[:, 2], bins=50, alpha=0.7, label="Scale Z")
            else:
                axes[0, 2].hist(scales.flatten(), bins=50, alpha=0.7)
            axes[0, 2].set_title("Scale Distribution")
            axes[0, 2].legend()
        
        # Opacity distribution
        if "opacity" in gaussians:
            opacity = gaussians["opacity"]
            # Convert from logit space
            opacity_prob = 1 / (1 + np.exp(-opacity))
            axes[1, 0].hist(opacity_prob.flatten(), bins=50, alpha=0.7)
            axes[1, 0].set_title("Opacity Distribution")
        
        # 3D scatter plot of positions
        ax_3d = axes[1, 1]
        ax_3d.remove()
        ax_3d = fig.add_subplot(2, 3, 5, projection='3d')
        
        # Sample points for visualization
        n_sample = min(1000, len(positions))
        indices = np.random.choice(len(positions), n_sample, replace=False)
        sample_pos = positions[indices]
        
        if "colors" in gaussians:
            sample_colors = gaussians["colors"][indices]
            ax_3d.scatter(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2], 
                         c=sample_colors, s=1)
        else:
            ax_3d.scatter(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2], s=1)
        
        ax_3d.set_title("3D Position Scatter")
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        
        # Summary statistics
        axes[1, 2].axis('off')
        stats_text = f"""
        Gaussian Statistics:
        Total Gaussians: {len(positions):,}
        
        Position Range:
        X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]
        Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]
        Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]
        """
        
        if "opacity" in gaussians:
            opacity_prob = 1 / (1 + np.exp(-gaussians["opacity"]))
            stats_text += f"""
        Opacity:
        Mean: {opacity_prob.mean():.3f}
        Std: {opacity_prob.std():.3f}
        """
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Gaussian statistics plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_temporal_statistics(self, 
                               correspondences_data: Dict[str, Any],
                               save_path: str = None) -> None:
        """Plot temporal correspondence statistics"""
        correspondences = correspondences_data["temporal_correspondences"]
        
        # Extract statistics
        frame_pairs = []
        num_correspondences = []
        avg_distances = []
        
        for corr in correspondences:
            frame_pairs.append(f"{corr['frame1']}-{corr['frame2']}")
            num_correspondences.append(corr['num_correspondences'])
            
            if corr['correspondences']:
                distances = [c['distance'] for c in corr['correspondences']]
                avg_distances.append(np.mean(distances))
            else:
                avg_distances.append(0)
        
        # Create plots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("Temporal Correspondence Statistics", fontsize=16)
        
        # Number of correspondences
        axes[0].bar(range(len(frame_pairs)), num_correspondences)
        axes[0].set_title("Number of Correspondences Between Frames")
        axes[0].set_xlabel("Frame Pairs")
        axes[0].set_ylabel("Number of Correspondences")
        axes[0].tick_params(axis='x', rotation=45)
        
        # Average distances
        axes[1].plot(range(len(frame_pairs)), avg_distances, 'o-')
        axes[1].set_title("Average Correspondence Distance")
        axes[1].set_xlabel("Frame Pairs")
        axes[1].set_ylabel("Average Distance")
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Temporal statistics plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_visualization_summary(self, 
                                   video_name: str,
                                   gaussian_dir: str) -> None:
        """Create comprehensive visualization summary"""
        self.logger.info(f"Creating visualization summary for {video_name}")
        
        gaussian_path = Path(gaussian_dir) / video_name
        viz_dir = self.output_dir / video_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and visualize first few frames
        gaussian_files = sorted(list(gaussian_path.glob("*_gaussians.npz")))
        
        for i, gaussian_file in enumerate(gaussian_files[:5]):  # First 5 frames
            data = np.load(gaussian_file)
            
            # Convert data to dict
            gaussians = {}
            for key in data.files:
                if key != "metadata":
                    gaussians[key] = data[key]
            
            # Plot statistics
            stats_path = viz_dir / f"frame_{i:03d}_statistics.png"
            self.plot_gaussian_statistics(gaussians, str(stats_path))
            
            # Save point cloud if positions and colors available
            if "positions" in gaussians and "colors" in gaussians:
                positions = gaussians["positions"]
                colors = gaussians["colors"]
                point_cloud = np.concatenate([positions, colors], axis=1)
                
                pc_path = viz_dir / f"frame_{i:03d}_pointcloud.ply"
                self.visualize_point_cloud(point_cloud, save_path=str(pc_path))
        
        # Load and visualize temporal correspondences
        corr_file = gaussian_path / "temporal_correspondences.json"
        if corr_file.exists():
            with open(corr_file, 'r') as f:
                correspondences = json.load(f)
            
            corr_path = viz_dir / "temporal_correspondences.png"
            self.plot_temporal_statistics(correspondences, str(corr_path))
        
        self.logger.info(f"Visualization summary created in {viz_dir}")
