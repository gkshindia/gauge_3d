"""
4D Gaussian Splatting Visualization Tools

Provides tools to visualize:
- Individual Gaussian splats
- Temporal Gaussian evolution
- Gaussian correspondence tracking
- 3D point cloud representations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from pathlib import Path
import json
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm
import cv2


class GaussianViewer:
    """Visualization tools for 4D Gaussian splatting outputs."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.gaussian_dir = self.output_dir / "gaussian_reconstruction"
        self.gaussian_init_dir = self.gaussian_dir / "gaussian_init"
        self.point_clouds_dir = self.gaussian_dir / "point_clouds"
        self.logs_dir = self.gaussian_dir / "logs"
        
    def load_gaussian_frame(self, dataset_name: str, frame_idx: int) -> Optional[Dict]:
        """Load Gaussian data for a specific frame."""
        gaussian_file = self.gaussian_init_dir / dataset_name / f"frame_{frame_idx:06d}_gaussians.npz"
        if gaussian_file.exists():
            data = np.load(gaussian_file)
            return {key: data[key] for key in data.files}
        return None
    
    def load_point_cloud(self, dataset_name: str, frame_idx: int) -> Optional[Dict]:
        """Load point cloud data for a specific frame."""
        pc_file = self.point_clouds_dir / dataset_name / f"frame_{frame_idx:06d}_pointcloud.npz"
        if pc_file.exists():
            data = np.load(pc_file)
            return {key: data[key] for key in data.files}
        return None
    
    def visualize_gaussian_positions(self, dataset_name: str, frame_idx: int, 
                                   sample_ratio: float = 0.1) -> plt.Figure:
        """Visualize 3D positions of Gaussians."""
        gaussian_data = self.load_gaussian_frame(dataset_name, frame_idx)
        if not gaussian_data:
            print(f"No Gaussian data found for frame {frame_idx}")
            return None
        
        positions = gaussian_data.get('positions')
        colors = gaussian_data.get('colors')
        
        if positions is None:
            print("No position data found")
            return None
        
        # Sample points for better visualization
        num_points = len(positions)
        sample_size = max(1, int(num_points * sample_ratio))
        indices = np.random.choice(num_points, sample_size, replace=False)
        
        sampled_positions = positions[indices]
        sampled_colors = colors[indices] if colors is not None else None
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if sampled_colors is not None:
            # Normalize colors to [0, 1] range
            if sampled_colors.max() > 1.0:
                sampled_colors = sampled_colors / 255.0
            
            ax.scatter(sampled_positions[:, 0], sampled_positions[:, 1], sampled_positions[:, 2], 
                      c=sampled_colors, s=1, alpha=0.6)
        else:
            ax.scatter(sampled_positions[:, 0], sampled_positions[:, 1], sampled_positions[:, 2], 
                      s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Gaussian Positions - Frame {frame_idx:06d}\n'
                    f'({sample_size:,} / {num_points:,} points)')
        
        return fig
    
    def visualize_gaussian_sequence(self, dataset_name: str, frame_indices: List[int], 
                                  sample_ratio: float = 0.05) -> plt.Figure:
        """Visualize Gaussian evolution over multiple frames."""
        num_frames = len(frame_indices)
        fig = plt.figure(figsize=(5*num_frames, 5))
        
        for i, frame_idx in enumerate(frame_indices):
            gaussian_data = self.load_gaussian_frame(dataset_name, frame_idx)
            if not gaussian_data:
                continue
                
            positions = gaussian_data.get('positions')
            colors = gaussian_data.get('colors')
            
            if positions is None:
                continue
            
            # Sample points
            num_points = len(positions)
            sample_size = max(1, int(num_points * sample_ratio))
            indices = np.random.choice(num_points, sample_size, replace=False)
            
            sampled_positions = positions[indices]
            sampled_colors = colors[indices] if colors is not None else None
            
            ax = fig.add_subplot(1, num_frames, i+1, projection='3d')
            
            if sampled_colors is not None:
                if sampled_colors.max() > 1.0:
                    sampled_colors = sampled_colors / 255.0
                ax.scatter(sampled_positions[:, 0], sampled_positions[:, 1], sampled_positions[:, 2], 
                          c=sampled_colors, s=2, alpha=0.7)
            else:
                ax.scatter(sampled_positions[:, 0], sampled_positions[:, 1], sampled_positions[:, 2], 
                          s=2, alpha=0.7)
            
            ax.set_title(f'Frame {frame_idx:06d}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.tight_layout()
        return fig
    
    def visualize_gaussian_properties(self, dataset_name: str, frame_idx: int) -> plt.Figure:
        """Visualize various Gaussian properties."""
        gaussian_data = self.load_gaussian_frame(dataset_name, frame_idx)
        if not gaussian_data:
            print(f"No Gaussian data found for frame {frame_idx}")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Position distribution
        positions = gaussian_data.get('positions')
        if positions is not None:
            axes[0, 0].hist(positions[:, 0], bins=50, alpha=0.7, label='X')
            axes[0, 0].hist(positions[:, 1], bins=50, alpha=0.7, label='Y')
            axes[0, 0].hist(positions[:, 2], bins=50, alpha=0.7, label='Z')
            axes[0, 0].set_title('Position Distribution')
            axes[0, 0].set_xlabel('Position')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Color distribution
        colors = gaussian_data.get('colors')
        if colors is not None:
            if colors.shape[1] >= 3:
                axes[0, 1].hist(colors[:, 0], bins=50, alpha=0.7, label='R', color='red')
                axes[0, 1].hist(colors[:, 1], bins=50, alpha=0.7, label='G', color='green')
                axes[0, 1].hist(colors[:, 2], bins=50, alpha=0.7, label='B', color='blue')
                axes[0, 1].set_title('Color Distribution')
                axes[0, 1].set_xlabel('Color Value')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
        
        # Scale/Size distribution
        scales = gaussian_data.get('scales')
        if scales is not None:
            if scales.ndim > 1:
                for i in range(min(3, scales.shape[1])):
                    axes[0, 2].hist(scales[:, i], bins=50, alpha=0.7, label=f'Scale {i}')
            else:
                axes[0, 2].hist(scales, bins=50, alpha=0.7)
            axes[0, 2].set_title('Scale Distribution')
            axes[0, 2].set_xlabel('Scale')
            axes[0, 2].set_ylabel('Count')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Opacity distribution
        opacities = gaussian_data.get('opacities')
        if opacities is not None:
            axes[1, 0].hist(opacities, bins=50, alpha=0.7)
            axes[1, 0].set_title('Opacity Distribution')
            axes[1, 0].set_xlabel('Opacity')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Correspondences (if available)
        correspondences = gaussian_data.get('correspondences')
        if correspondences is not None:
            valid_correspondences = correspondences[correspondences >= 0]
            if len(valid_correspondences) > 0:
                axes[1, 1].hist(valid_correspondences, bins=50, alpha=0.7)
                axes[1, 1].set_title('Correspondence Distribution')
                axes[1, 1].set_xlabel('Correspondence ID')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"Frame {frame_idx:06d} Statistics:\n\n"
        if positions is not None:
            stats_text += f"Total Gaussians: {len(positions):,}\n"
            stats_text += f"Position Range:\n"
            stats_text += f"  X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]\n"
            stats_text += f"  Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]\n"
            stats_text += f"  Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]\n\n"
        
        if colors is not None:
            stats_text += f"Color Range: [{colors.min():.2f}, {colors.max():.2f}]\n"
        
        if correspondences is not None:
            valid_corr = correspondences[correspondences >= 0]
            stats_text += f"Valid Correspondences: {len(valid_corr):,} / {len(correspondences):,}\n"
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Summary Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_gaussian_video(self, dataset_name: str, output_path: Optional[str] = None,
                            fps: int = 10, sample_ratio: float = 0.02) -> str:
        """Create a video showing Gaussian evolution over time."""
        gaussian_init_dir = self.gaussian_init_dir / dataset_name
        if not gaussian_init_dir.exists():
            raise ValueError(f"Gaussian directory not found: {gaussian_init_dir}")
        
        if output_path is None:
            output_path = self.gaussian_dir / f"{dataset_name}_gaussian_video.mp4"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all Gaussian files
        gaussian_files = sorted(list(gaussian_init_dir.glob("frame_*_gaussians.npz")))
        if not gaussian_files:
            raise ValueError(f"No Gaussian files found in {gaussian_init_dir}")
        
        print(f"Creating Gaussian video from {len(gaussian_files)} frames...")
        
        # Process frames and save as images first
        temp_dir = output_path.parent / "temp_gaussian_frames"
        temp_dir.mkdir(exist_ok=True)
        
        for i, gaussian_file in enumerate(tqdm(gaussian_files)):
            frame_idx = int(gaussian_file.stem.split('_')[1])
            
            fig = self.visualize_gaussian_positions(dataset_name, frame_idx, sample_ratio)
            if fig:
                temp_file = temp_dir / f"frame_{i:06d}.png"
                fig.savefig(temp_file, dpi=100, bbox_inches='tight')
                plt.close(fig)
        
        # Create video from saved frames
        frame_files = sorted(list(temp_dir.glob("frame_*.png")))
        if frame_files:
            # Read first frame to get dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            height, width = first_frame.shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                video_writer.write(frame)
            
            video_writer.release()
        
        # Clean up temporary files
        for temp_file in temp_dir.glob("*.png"):
            temp_file.unlink()
        temp_dir.rmdir()
        
        print(f"Gaussian video saved to: {output_path}")
        return str(output_path)
    
    def export_to_open3d(self, dataset_name: str, frame_idx: int) -> Optional[o3d.geometry.PointCloud]:
        """Export Gaussian data to Open3D point cloud for interactive viewing."""
        gaussian_data = self.load_gaussian_frame(dataset_name, frame_idx)
        if not gaussian_data:
            return None
        
        positions = gaussian_data.get('positions')
        colors = gaussian_data.get('colors')
        
        if positions is None:
            return None
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        
        if colors is not None:
            # Normalize colors to [0, 1] range
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        return pcd
    
    def interactive_view(self, dataset_name: str, frame_idx: int):
        """Launch interactive Open3D viewer."""
        pcd = self.export_to_open3d(dataset_name, frame_idx)
        if pcd:
            o3d.visualization.draw_geometries([pcd], 
                                            window_name=f"Gaussians - {dataset_name} Frame {frame_idx:06d}")
        else:
            print(f"Could not load data for frame {frame_idx}")
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available Gaussian datasets."""
        if not self.gaussian_init_dir.exists():
            return []
        
        datasets = [d.name for d in self.gaussian_init_dir.iterdir() if d.is_dir()]
        return sorted(datasets)
    
    def get_frame_count(self, dataset_name: str) -> int:
        """Get the number of frames in a Gaussian dataset."""
        gaussian_dir = self.gaussian_init_dir / dataset_name
        if not gaussian_dir.exists():
            return 0
        
        gaussian_files = list(gaussian_dir.glob("frame_*_gaussians.npz"))
        return len(gaussian_files)
    
    def analyze_temporal_consistency(self, dataset_name: str, max_frames: int = 100) -> Dict:
        """Analyze temporal consistency of Gaussians across frames."""
        gaussian_dir = self.gaussian_init_dir / dataset_name
        if not gaussian_dir.exists():
            return {}
        
        gaussian_files = sorted(list(gaussian_dir.glob("frame_*_gaussians.npz")))[:max_frames]
        
        frame_stats = {}
        total_gaussians = []
        position_ranges = []
        
        for gaussian_file in tqdm(gaussian_files, desc="Analyzing frames"):
            frame_idx = int(gaussian_file.stem.split('_')[1])
            
            data = np.load(gaussian_file)
            positions = data.get('positions')
            
            if positions is not None:
                stats = {
                    'num_gaussians': len(positions),
                    'position_mean': positions.mean(axis=0).tolist(),
                    'position_std': positions.std(axis=0).tolist(),
                    'position_range': [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()]
                }
                frame_stats[f"frame_{frame_idx:06d}"] = stats
                total_gaussians.append(len(positions))
                
                pos_range = positions.max(axis=0) - positions.min(axis=0)
                position_ranges.append(pos_range.tolist())
        
        return {
            'frame_stats': frame_stats,
            'total_frames_analyzed': len(frame_stats),
            'gaussian_count_stats': {
                'mean': np.mean(total_gaussians),
                'std': np.std(total_gaussians),
                'min': np.min(total_gaussians),
                'max': np.max(total_gaussians)
            },
            'position_range_stats': {
                'mean': np.mean(position_ranges, axis=0).tolist(),
                'std': np.std(position_ranges, axis=0).tolist()
            }
        }


def main():
    """Demo function to test Gaussian visualization."""
    viewer = GaussianViewer()
    datasets = viewer.get_available_datasets()
    
    if not datasets:
        print("No Gaussian datasets found!")
        return
    
    print(f"Available datasets: {datasets}")
    
    # Visualize the first dataset
    dataset = datasets[0]
    print(f"\nVisualizing dataset: {dataset}")
    
    frame_count = viewer.get_frame_count(dataset)
    print(f"Total frames: {frame_count}")
    
    if frame_count > 0:
        # Show Gaussian properties for first frame
        fig1 = viewer.visualize_gaussian_properties(dataset, 0)
        if fig1:
            plt.show()
        
        # Show sequence of frames
        frame_indices = list(range(0, min(5, frame_count)))
        fig2 = viewer.visualize_gaussian_sequence(dataset, frame_indices)
        plt.show()
        
        # Interactive view (uncomment to launch)
        # viewer.interactive_view(dataset, 0)


if __name__ == "__main__":
    main()
