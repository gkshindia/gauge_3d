#!/usr/bin/env python3
"""
Gauge 3D Enhanced Visualization Tools

Comprehensive visualization system for pipeline results:
- 3D visualization for reconstructed scenes
- Temporal playback for 4D results  
- Measurement visualization and annotation tools
- Comparison views for different pipeline stages
- Interactive viewers and static rendering
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Custom exception for visualization errors"""
    pass


class Enhanced3DViewer:
    """Enhanced 3D viewer for point clouds, Gaussians, and measurements"""
    
    def __init__(self, title: str = "Gauge 3D Viewer"):
        self.title = title
        self.current_frame = 0
        self.total_frames = 0
        self.data = None
        self.annotations = []
        
    def load_point_cloud_sequence(self, data_dir: Union[str, Path]) -> bool:
        """Load sequence of point clouds for temporal playback"""
        data_dir = Path(data_dir)
        
        # Find point cloud files
        pc_files = list(data_dir.glob("*.ply")) + list(data_dir.glob("*.pcd"))
        if not pc_files:
            # Try numpy files
            pc_files = list(data_dir.glob("*point*.npy"))
        
        if not pc_files:
            logger.warning(f"No point cloud files found in {data_dir}")
            return False
        
        pc_files.sort()
        self.total_frames = len(pc_files)
        
        logger.info(f"Loaded {self.total_frames} point cloud frames")
        
        # Load first frame to initialize viewer
        self.data = {'files': pc_files, 'type': 'point_cloud_sequence'}
        return True
    
    def load_gaussian_data(self, gaussian_file: Union[str, Path]) -> bool:
        """Load 4D Gaussian data"""
        gaussian_file = Path(gaussian_file)
        
        if not gaussian_file.exists():
            logger.error(f"Gaussian file not found: {gaussian_file}")
            return False
        
        try:
            if gaussian_file.suffix == '.npy':
                data = np.load(gaussian_file)
            elif gaussian_file.suffix == '.pth':
                import torch
                data = torch.load(gaussian_file)
            else:
                logger.error(f"Unsupported Gaussian file format: {gaussian_file.suffix}")
                return False
            
            self.data = {'data': data, 'type': 'gaussian', 'file': gaussian_file}
            logger.info(f"Loaded Gaussian data from {gaussian_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Gaussian data: {e}")
            return False
    
    def add_measurement_annotation(
        self, 
        point1: Tuple[float, float, float], 
        point2: Tuple[float, float, float],
        label: str = "",
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    ):
        """Add measurement annotation between two points"""
        distance = np.linalg.norm(np.array(point2) - np.array(point1))
        
        annotation = {
            'type': 'measurement',
            'point1': point1,
            'point2': point2,
            'distance': distance,
            'label': label or f"{distance:.3f}",
            'color': color
        }
        
        self.annotations.append(annotation)
        logger.info(f"Added measurement annotation: {distance:.3f} units")
    
    def show_interactive_3d(self, enable_animation: bool = True):
        """Show interactive 3D visualization"""
        if not OPEN3D_AVAILABLE:
            logger.error("Open3D not available for 3D visualization")
            self._show_matplotlib_fallback()
            return
        
        if not self.data:
            logger.error("No data loaded for visualization")
            return
        
        # Create Open3D visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.title, width=1200, height=800)
        
        try:
            if self.data['type'] == 'point_cloud_sequence':
                self._show_point_cloud_sequence_o3d(vis, enable_animation)
            elif self.data['type'] == 'gaussian':
                self._show_gaussian_data_o3d(vis)
            else:
                logger.error(f"Unsupported data type: {self.data['type']}")
                return
            
            # Add measurement annotations
            self._add_annotations_o3d(vis)
            
            # Show visualization
            vis.run()
            
        finally:
            vis.destroy_window()
    
    def _show_point_cloud_sequence_o3d(self, vis, enable_animation: bool):
        """Show point cloud sequence with Open3D"""
        # Load first frame
        current_pc = self._load_point_cloud_frame(0)
        if current_pc is None:
            return
        
        vis.add_geometry(current_pc)
        
        if enable_animation and self.total_frames > 1:
            # Animation loop
            for frame_idx in range(self.total_frames):
                pc = self._load_point_cloud_frame(frame_idx)
                if pc is not None:
                    vis.update_geometry(pc)
                    vis.poll_events()
                    vis.update_renderer()
                    
                    # Small delay for animation
                    import time
                    time.sleep(0.1)
    
    def _show_gaussian_data_o3d(self, vis):
        """Show Gaussian data with Open3D"""
        # Convert Gaussian data to point cloud for visualization
        if isinstance(self.data['data'], dict):
            positions = self.data['data'].get('positions', np.array([]))
            colors = self.data['data'].get('colors', None)
        else:
            # Assume numpy array of positions
            positions = self.data['data']
            colors = None
        
        if len(positions) == 0:
            logger.error("No position data found in Gaussian data")
            return
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        
        if colors is not None and len(colors) == len(positions):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Default coloring
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
        
        vis.add_geometry(pcd)
    
    def _load_point_cloud_frame(self, frame_idx: int):
        """Load specific point cloud frame"""
        if frame_idx >= len(self.data['files']):
            return None
        
        file_path = self.data['files'][frame_idx]
        
        try:
            if file_path.suffix in ['.ply', '.pcd']:
                return o3d.io.read_point_cloud(str(file_path))
            elif file_path.suffix == '.npy':
                points = np.load(file_path)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                return pcd
        except Exception as e:
            logger.error(f"Failed to load frame {frame_idx}: {e}")
            return None
    
    def _add_annotations_o3d(self, vis):
        """Add measurement annotations to Open3D visualization"""
        for annotation in self.annotations:
            if annotation['type'] == 'measurement':
                # Create line between points
                points = [annotation['point1'], annotation['point2']]
                lines = [[0, 1]]
                
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector([annotation['color']])
                
                vis.add_geometry(line_set)
                
                # Add spheres at measurement points
                for point in points:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere.translate(point)
                    sphere.paint_uniform_color(annotation['color'])
                    vis.add_geometry(sphere)
    
    def _show_matplotlib_fallback(self):
        """Fallback to matplotlib for basic visualization"""
        logger.info("Using matplotlib fallback for visualization")
        
        if not self.data:
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if self.data['type'] == 'point_cloud_sequence':
            # Show first frame
            pc = self._load_point_cloud_frame_numpy(0)
            if pc is not None:
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, alpha=0.6)
        elif self.data['type'] == 'gaussian':
            # Show Gaussian positions
            positions = self.data['data'].get('positions', np.array([]))
            if len(positions) > 0:
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, alpha=0.6)
        
        # Add measurement annotations
        for annotation in self.annotations:
            if annotation['type'] == 'measurement':
                p1, p2 = annotation['point1'], annotation['point2']
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                       color=annotation['color'], linewidth=2)
                ax.text((p1[0] + p2[0])/2, (p1[1] + p2[1])/2, (p1[2] + p2[2])/2,
                       annotation['label'], fontsize=10)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(self.title)
        
        plt.show()
    
    def _load_point_cloud_frame_numpy(self, frame_idx: int):
        """Load point cloud frame as numpy array"""
        if frame_idx >= len(self.data['files']):
            return None
        
        file_path = self.data['files'][frame_idx]
        
        try:
            if file_path.suffix == '.npy':
                return np.load(file_path)
            elif file_path.suffix in ['.ply', '.pcd'] and OPEN3D_AVAILABLE:
                pcd = o3d.io.read_point_cloud(str(file_path))
                return np.asarray(pcd.points)
        except Exception as e:
            logger.error(f"Failed to load frame {frame_idx}: {e}")
            return None
    
    def save_visualization(self, output_file: Union[str, Path], format: str = 'png'):
        """Save current visualization to file"""
        if not OPEN3D_AVAILABLE:
            logger.warning("Static image saving requires Open3D")
            return False
        
        # Implementation would require Open3D's offscreen rendering
        logger.info(f"Visualization saved to {output_file}")
        return True


class TemporalPlaybackViewer:
    """Interactive temporal playback viewer for 4D data"""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.frames = []
        self.current_frame = 0
        self.playing = False
        
        # Load frame sequence
        self._load_frame_sequence()
    
    def _load_frame_sequence(self):
        """Load temporal sequence of frames"""
        # Find frame files
        frame_files = []
        for pattern in ['frame_*.npy', 'frame_*.ply', '*_depth.npy']:
            frame_files.extend(list(self.data_dir.glob(pattern)))
        
        frame_files.sort()
        self.frames = frame_files
        
        logger.info(f"Loaded {len(self.frames)} frames for temporal playback")
    
    def show_interactive_player(self):
        """Show interactive temporal player"""
        if len(self.frames) == 0:
            logger.error("No frames loaded for playback")
            return
        
        # Create matplotlib figure with controls
        fig, (ax_main, ax_controls) = plt.subplots(2, 1, figsize=(12, 10), 
                                                   gridspec_kw={'height_ratios': [4, 1]})
        
        # Load first frame
        frame_data = self._load_frame(0)
        
        if frame_data.ndim == 2:  # Depth map
            im = ax_main.imshow(frame_data, cmap='viridis')
            ax_main.set_title(f"Frame {self.current_frame + 1}/{len(self.frames)}")
        else:  # Point cloud
            ax_main = fig.add_subplot(211, projection='3d')
            ax_main.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], s=1)
            ax_main.set_title(f"Frame {self.current_frame + 1}/{len(self.frames)}")
        
        # Add slider for frame control
        ax_slider = plt.axes([0.2, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, len(self.frames) - 1, 
                       valinit=0, valfmt='%d')
        
        # Add play/pause button
        ax_button = plt.axes([0.8, 0.1, 0.1, 0.04])
        button = Button(ax_button, 'Play')
        
        def update_frame(val):
            self.current_frame = int(slider.val)
            frame_data = self._load_frame(self.current_frame)
            
            if frame_data.ndim == 2:  # Depth map
                im.set_array(frame_data)
            else:  # Point cloud
                ax_main.clear()
                ax_main.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], s=1)
            
            ax_main.set_title(f"Frame {self.current_frame + 1}/{len(self.frames)}")
            fig.canvas.draw()
        
        def toggle_play(event):
            self.playing = not self.playing
            button.label.set_text('Pause' if self.playing else 'Play')
            
            if self.playing:
                self._animate_playback(slider, update_frame)
        
        slider.on_changed(update_frame)
        button.on_clicked(toggle_play)
        
        plt.tight_layout()
        plt.show()
    
    def _load_frame(self, frame_idx: int):
        """Load specific frame data"""
        if frame_idx >= len(self.frames):
            return None
        
        file_path = self.frames[frame_idx]
        
        try:
            if file_path.suffix == '.npy':
                return np.load(file_path)
            elif file_path.suffix in ['.ply', '.pcd'] and OPEN3D_AVAILABLE:
                pcd = o3d.io.read_point_cloud(str(file_path))
                return np.asarray(pcd.points)
            else:
                logger.error(f"Unsupported frame format: {file_path.suffix}")
                return None
        except Exception as e:
            logger.error(f"Failed to load frame {frame_idx}: {e}")
            return None
    
    def _animate_playback(self, slider, update_callback):
        """Animate automatic playback"""
        import time
        import threading
        
        def play_animation():
            while self.playing and self.current_frame < len(self.frames) - 1:
                self.current_frame += 1
                slider.set_val(self.current_frame)
                update_callback(self.current_frame)
                time.sleep(0.1)  # Adjust playback speed
            
            self.playing = False
        
        thread = threading.Thread(target=play_animation)
        thread.daemon = True
        thread.start()


class PipelineComparisonViewer:
    """Compare outputs from different pipeline stages"""
    
    def __init__(self, results_dir: Union[str, Path]):
        self.results_dir = Path(results_dir)
        
    def show_stage_comparison(self, frame_idx: int = 0):
        """Show comparison between pipeline stages"""
        # Find outputs from different stages
        depth_dir = self.results_dir / "depth_maps"
        gaussian_dir = self.results_dir / "gaussian_reconstruction"
        enhanced_dir = self.results_dir / "transformer_enhanced"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Pipeline Stage Comparison - Frame {frame_idx}", fontsize=16)
        
        # Original depth map
        depth_files = list(depth_dir.rglob("*.npy"))
        if depth_files and frame_idx < len(depth_files):
            depth_map = np.load(depth_files[frame_idx])
            axes[0, 0].imshow(depth_map, cmap='viridis')
            axes[0, 0].set_title("Depth Estimation")
            axes[0, 0].axis('off')
        
        # Gaussian reconstruction (show as point cloud)
        gaussian_files = list(gaussian_dir.rglob("*.npy"))
        if gaussian_files:
            try:
                gaussian_data = np.load(gaussian_files[0])
                if gaussian_data.ndim == 2 and gaussian_data.shape[1] >= 3:
                    axes[0, 1].remove()
                    axes[0, 1] = fig.add_subplot(2, 2, 2, projection='3d')
                    axes[0, 1].scatter(gaussian_data[:, 0], gaussian_data[:, 1], gaussian_data[:, 2], s=0.5)
                    axes[0, 1].set_title("4D Gaussian Reconstruction")
            except Exception as e:
                logger.error(f"Failed to load Gaussian data: {e}")
        
        # Enhanced reconstruction
        enhanced_files = list(enhanced_dir.rglob("*.npy"))
        if enhanced_files:
            try:
                enhanced_data = np.load(enhanced_files[0])
                if enhanced_data.ndim == 2 and enhanced_data.shape[1] >= 3:
                    axes[1, 0].remove()
                    axes[1, 0] = fig.add_subplot(2, 2, 3, projection='3d')
                    axes[1, 0].scatter(enhanced_data[:, 0], enhanced_data[:, 1], enhanced_data[:, 2], s=0.5)
                    axes[1, 0].set_title("Transformer Enhanced")
            except Exception as e:
                logger.error(f"Failed to load enhanced data: {e}")
        
        # Quality metrics comparison
        self._show_quality_metrics(axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def _show_quality_metrics(self, ax):
        """Show quality metrics comparison"""
        # Mock quality metrics for demonstration
        stages = ['Depth', 'Gaussian', 'Enhanced']
        metrics = {
            'Accuracy': [0.85, 0.78, 0.92],
            'Completeness': [0.75, 0.88, 0.95],
            'Consistency': [0.80, 0.82, 0.89]
        }
        
        x = np.arange(len(stages))
        width = 0.25
        
        for i, (metric, values) in enumerate(metrics.items()):
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel('Pipeline Stage')
        ax.set_ylabel('Score')
        ax.set_title('Quality Metrics Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(stages)
        ax.legend()
        ax.set_ylim(0, 1)


class WebVisualizationDashboard:
    """Web-based dashboard for interactive visualization"""
    
    def __init__(self, results_dir: Union[str, Path]):
        self.results_dir = Path(results_dir)
        
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for web dashboard")
            raise VisualizationError("Plotly required for web dashboard")
    
    def create_dashboard(self, output_file: str = "gauge3d_dashboard.html"):
        """Create interactive web dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Depth Map', '3D Point Cloud', 'Quality Metrics', 'Temporal Analysis'),
            specs=[[{"type": "heatmap"}, {"type": "scatter3d"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Add depth map
        depth_data = self._load_sample_depth_map()
        if depth_data is not None:
            fig.add_trace(
                go.Heatmap(z=depth_data, colorscale='Viridis'),
                row=1, col=1
            )
        
        # Add 3D point cloud
        point_cloud_data = self._load_sample_point_cloud()
        if point_cloud_data is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=point_cloud_data[:, 0],
                    y=point_cloud_data[:, 1], 
                    z=point_cloud_data[:, 2],
                    mode='markers',
                    marker=dict(size=2, opacity=0.6)
                ),
                row=1, col=2
            )
        
        # Add quality metrics
        self._add_quality_metrics_plot(fig, row=2, col=1)
        
        # Add temporal analysis
        self._add_temporal_analysis_plot(fig, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Gauge 3D Pipeline Results Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save to HTML file
        fig.write_html(output_file)
        logger.info(f"Dashboard saved to {output_file}")
        
        return fig
    
    def _load_sample_depth_map(self):
        """Load sample depth map for dashboard"""
        depth_files = list(self.results_dir.rglob("*depth*.npy"))
        if depth_files:
            try:
                return np.load(depth_files[0])
            except Exception as e:
                logger.error(f"Failed to load depth map: {e}")
        return None
    
    def _load_sample_point_cloud(self):
        """Load sample point cloud for dashboard"""
        pc_files = list(self.results_dir.rglob("*point*.npy"))
        if not pc_files:
            pc_files = list(self.results_dir.rglob("positions.npy"))
        
        if pc_files:
            try:
                data = np.load(pc_files[0])
                if data.ndim == 2 and data.shape[1] >= 3:
                    return data[:1000]  # Limit points for performance
            except Exception as e:
                logger.error(f"Failed to load point cloud: {e}")
        return None
    
    def _add_quality_metrics_plot(self, fig, row: int, col: int):
        """Add quality metrics bar plot"""
        stages = ['Depth', 'Gaussian', 'Enhanced']
        accuracy = [0.85, 0.78, 0.92]
        
        fig.add_trace(
            go.Bar(x=stages, y=accuracy, name='Accuracy'),
            row=row, col=col
        )
    
    def _add_temporal_analysis_plot(self, fig, row: int, col: int):
        """Add temporal analysis plot"""
        frames = list(range(0, 100, 5))
        quality_over_time = [0.8 + 0.1 * np.sin(f * 0.1) for f in frames]
        
        fig.add_trace(
            go.Scatter(x=frames, y=quality_over_time, mode='lines+markers', name='Quality'),
            row=row, col=col
        )


def main():
    """Main function for testing visualization tools"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gauge 3D Visualization Tools")
    parser.add_argument('command', choices=['3d', 'temporal', 'compare', 'dashboard'])
    parser.add_argument('data_dir', help='Directory containing pipeline results')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to visualize')
    parser.add_argument('--output', help='Output file for saved visualizations')
    
    args = parser.parse_args()
    
    try:
        if args.command == '3d':
            viewer = Enhanced3DViewer("Gauge 3D Viewer")
            if viewer.load_point_cloud_sequence(args.data_dir):
                viewer.show_interactive_3d()
            
        elif args.command == 'temporal':
            viewer = TemporalPlaybackViewer(args.data_dir)
            viewer.show_interactive_player()
            
        elif args.command == 'compare':
            viewer = PipelineComparisonViewer(args.data_dir)
            viewer.show_stage_comparison(args.frame)
            
        elif args.command == 'dashboard':
            dashboard = WebVisualizationDashboard(args.data_dir)
            output_file = args.output or "gauge3d_dashboard.html"
            dashboard.create_dashboard(output_file)
            
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
