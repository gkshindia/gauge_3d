"""
Point Cloud Visualization Module

Provides tools for visualizing and analyzing point clouds generated from the 3D reconstruction pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Union
import logging

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logging.warning("Open3D not available. 3D visualization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logging.warning("Plotly not available. Interactive 3D visualization will be limited.")


class PointCloudViewer:
    """Comprehensive point cloud visualization and analysis tools."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the PointCloudViewer.
        
        Args:
            output_dir: Base output directory containing point cloud data
        """
        self.output_dir = Path(output_dir)
        self.point_cloud_dir = self.output_dir / "gaussian_reconstruction" / "point_clouds"
        
    def load_point_cloud(self, filepath: Union[str, Path]) -> Optional[Dict]:
        """
        Load a point cloud from file.
        
        Args:
            filepath: Path to the point cloud file (.ply, .pcd, .npz, .json)
            
        Returns:
            Dictionary containing point cloud data or None if failed
        """
        filepath = Path(filepath)
        
        try:
            if filepath.suffix == '.npz':
                data = np.load(filepath)
                return {
                    'points': data.get('points', data.get('positions')),
                    'colors': data.get('colors', None),
                    'normals': data.get('normals', None)
                }
            elif filepath.suffix == '.json':
                with open(filepath, 'r') as f:
                    return json.load(f)
            elif filepath.suffix in ['.ply', '.pcd'] and HAS_OPEN3D:
                pcd = o3d.io.read_point_cloud(str(filepath))
                return {
                    'points': np.asarray(pcd.points),
                    'colors': np.asarray(pcd.colors) if pcd.has_colors() else None,
                    'normals': np.asarray(pcd.normals) if pcd.has_normals() else None
                }
            else:
                logging.error(f"Unsupported file format: {filepath.suffix}")
                return None
                
        except Exception as e:
            logging.error(f"Failed to load point cloud {filepath}: {e}")
            return None
    
    def visualize_point_cloud_matplotlib(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                                       title: str = "Point Cloud", figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize point cloud using matplotlib 3D scatter plot.
        
        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3) or None
            title: Plot title
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors, s=1, alpha=0.6)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def visualize_point_cloud_plotly(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                                   title: str = "Point Cloud") -> Optional[go.Figure]:
        """
        Create interactive point cloud visualization using Plotly.
        
        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3) or None
            title: Plot title
            
        Returns:
            Plotly figure or None if Plotly not available
        """
        if not HAS_PLOTLY:
            logging.warning("Plotly not available for interactive visualization")
            return None
        
        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            color_array = colors
        else:
            color_array = points[:, 2]  # Color by Z coordinate
        
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1], 
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=color_array,
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        return fig
    
    def visualize_point_cloud_open3d(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                                    title: str = "Point Cloud"):
        """
        Visualize point cloud using Open3D.
        
        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3) or None
            title: Window title
        """
        if not HAS_OPEN3D:
            logging.warning("Open3D not available for 3D visualization")
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries([pcd], window_name=title)
    
    def analyze_point_cloud_statistics(self, points: np.ndarray) -> Dict:
        """
        Analyze point cloud statistics.
        
        Args:
            points: Point coordinates (N, 3)
            
        Returns:
            Dictionary containing statistics
        """
        stats = {
            'num_points': len(points),
            'bounds': {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist(),
                'range': (points.max(axis=0) - points.min(axis=0)).tolist()
            },
            'centroid': points.mean(axis=0).tolist(),
            'std': points.std(axis=0).tolist()
        }
        
        # Calculate density estimation
        if len(points) > 100:
            # Sample subset for density calculation to avoid memory issues
            sample_size = min(1000, len(points))
            sample_indices = np.random.choice(len(points), sample_size, replace=False)
            sample_points = points[sample_indices]
            
            # Calculate average nearest neighbor distance
            from scipy.spatial.distance import cdist
            distances = cdist(sample_points, sample_points)
            # Set diagonal to infinity to exclude self-distances
            np.fill_diagonal(distances, np.inf)
            nearest_distances = distances.min(axis=1)
            
            stats['density'] = {
                'avg_nearest_neighbor_distance': float(nearest_distances.mean()),
                'std_nearest_neighbor_distance': float(nearest_distances.std())
            }
        
        return stats
    
    def compare_point_clouds(self, cloud_paths: List[Union[str, Path]], 
                           labels: Optional[List[str]] = None) -> plt.Figure:
        """
        Compare multiple point clouds side by side.
        
        Args:
            cloud_paths: List of paths to point cloud files
            labels: Optional labels for each cloud
            
        Returns:
            Matplotlib figure with comparison plots
        """
        if labels is None:
            labels = [f"Cloud {i+1}" for i in range(len(cloud_paths))]
        
        n_clouds = len(cloud_paths)
        fig, axes = plt.subplots(2, (n_clouds + 1) // 2, figsize=(6 * n_clouds, 12))
        if n_clouds == 1:
            axes = [axes]
        axes = axes.flatten()
        
        stats_list = []
        
        for i, (cloud_path, label) in enumerate(zip(cloud_paths, labels)):
            cloud_data = self.load_point_cloud(cloud_path)
            if cloud_data is None:
                continue
                
            points = cloud_data['points']
            colors = cloud_data.get('colors')
            
            # Create 3D subplot
            ax = fig.add_subplot(2, (n_clouds + 1) // 2, i + 1, projection='3d')
            
            if colors is not None and len(colors) > 0:
                if colors.max() > 1.0:
                    colors = colors / 255.0
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=colors, s=0.5, alpha=0.6)
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=points[:, 2], cmap='viridis', s=0.5, alpha=0.6)
            
            ax.set_title(f"{label}\n{len(points):,} points")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Collect statistics
            stats = self.analyze_point_cloud_statistics(points)
            stats_list.append((label, stats))
        
        plt.tight_layout()
        
        # Print comparison statistics
        print("\nPoint Cloud Comparison Statistics:")
        print("=" * 50)
        for label, stats in stats_list:
            print(f"\n{label}:")
            print(f"  Points: {stats['num_points']:,}")
            print(f"  Bounds: X[{stats['bounds']['min'][0]:.2f}, {stats['bounds']['max'][0]:.2f}]")
            print(f"          Y[{stats['bounds']['min'][1]:.2f}, {stats['bounds']['max'][1]:.2f}]")
            print(f"          Z[{stats['bounds']['min'][2]:.2f}, {stats['bounds']['max'][2]:.2f}]")
            print(f"  Centroid: ({stats['centroid'][0]:.2f}, {stats['centroid'][1]:.2f}, {stats['centroid'][2]:.2f})")
        
        return fig
    
    def export_point_cloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                          output_path: Union[str, Path] = "point_cloud.ply"):
        """
        Export point cloud to file.
        
        Args:
            points: Point coordinates (N, 3)
            colors: Point colors (N, 3) or None
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        if output_path.suffix == '.ply' and HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            if colors is not None:
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            o3d.io.write_point_cloud(str(output_path), pcd)
            
        elif output_path.suffix == '.npz':
            save_dict = {'points': points}
            if colors is not None:
                save_dict['colors'] = colors
            np.savez_compressed(output_path, **save_dict)
            
        else:
            logging.error(f"Unsupported export format: {output_path.suffix}")
    
    def create_point_cloud_video(self, cloud_sequence: List[np.ndarray], 
                                output_path: Union[str, Path] = "point_cloud_evolution.mp4",
                                fps: int = 10, colors: Optional[List[np.ndarray]] = None):
        """
        Create a video showing point cloud evolution over time.
        
        Args:
            cloud_sequence: List of point cloud arrays
            output_path: Output video file path
            fps: Frames per second
            colors: Optional list of color arrays for each frame
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            logging.error("matplotlib.animation not available for video creation")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate global bounds
        all_points = np.vstack(cloud_sequence)
        bounds = {
            'x': [all_points[:, 0].min(), all_points[:, 0].max()],
            'y': [all_points[:, 1].min(), all_points[:, 1].max()],
            'z': [all_points[:, 2].min(), all_points[:, 2].max()]
        }
        
        def animate(frame):
            ax.clear()
            points = cloud_sequence[frame]
            
            if colors and frame < len(colors) and colors[frame] is not None:
                point_colors = colors[frame]
                if point_colors.max() > 1.0:
                    point_colors = point_colors / 255.0
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=point_colors, s=1, alpha=0.6)
            else:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
            
            ax.set_xlim(bounds['x'])
            ax.set_ylim(bounds['y'])
            ax.set_zlim(bounds['z'])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Point Cloud Evolution - Frame {frame + 1}/{len(cloud_sequence)}')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(cloud_sequence), 
                                     interval=1000//fps, blit=False)
        
        # Save video
        try:
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"Point cloud video saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save video: {e}")
            plt.show()
    
    def find_point_clouds(self, pattern: str = "*") -> List[Path]:
        """
        Find point cloud files in the output directory.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of point cloud file paths
        """
        extensions = ['.ply', '.pcd', '.npz', '.json']
        found_files = []
        
        for ext in extensions:
            found_files.extend(self.point_cloud_dir.glob(f"{pattern}{ext}"))
        
        return sorted(found_files)
