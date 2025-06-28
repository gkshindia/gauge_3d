#!/usr/bin/env python3
"""
Distance Measurement Implementation for Gauge 3D
Phase 4.2: 3D Point Selection and Distance Calculation

Implements interactive 3D point selection, distance calculation tools,
measurement validation, and export functionality for enhanced point clouds.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
    logger.warning("Open3D not available. 3D visualization will be limited.")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. 2D visualization will be limited.")


@dataclass
class MeasurementPoint:
    """Represents a 3D measurement point"""
    x: float
    y: float
    z: float
    frame_id: Optional[int] = None
    timestamp: Optional[str] = None
    confidence: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, **kwargs) -> 'MeasurementPoint':
        """Create from numpy array"""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]), **kwargs)


@dataclass
class DistanceMeasurement:
    """Represents a distance measurement between two points"""
    point_a: MeasurementPoint
    point_b: MeasurementPoint
    distance: float
    unit: str = "meters"
    measurement_id: Optional[str] = None
    timestamp: Optional[str] = None
    accuracy_estimate: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.measurement_id is None:
            self.measurement_id = f"meas_{int(datetime.now().timestamp())}"


class DistanceMeasurementTool:
    """Tool for measuring distances in 3D point clouds"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.measurements = []
        self.point_clouds = {}
        self.current_point_cloud = None
        
        logger.info("DistanceMeasurementTool initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for distance measurement"""
        return {
            "unit": "meters",
            "precision": 3,
            "snap_to_surface": True,
            "selection_tolerance": 0.01,
            "measurement_color": [1.0, 0.0, 0.0],
            "line_width": 2,
            "point_size": 5,
            "show_annotations": True,
            "enable_validation": True,
            "max_distance_threshold": 1000.0,
            "min_distance_threshold": 0.001
        }
    
    def load_point_cloud(
        self, 
        point_cloud_path: Union[str, Path],
        cloud_id: Optional[str] = None
    ) -> str:
        """
        Load point cloud for measurement
        
        Args:
            point_cloud_path: Path to point cloud file (.ply, .npy, etc.)
            cloud_id: Optional identifier for the point cloud
            
        Returns:
            Cloud ID string
        """
        point_cloud_path = Path(point_cloud_path)
        
        if cloud_id is None:
            cloud_id = point_cloud_path.stem
        
        logger.info(f"Loading point cloud: {point_cloud_path}")
        
        # Load point cloud based on file extension
        if point_cloud_path.suffix.lower() == '.ply':
            points = self._load_ply_file(point_cloud_path)
        elif point_cloud_path.suffix.lower() == '.npy':
            points = self._load_npy_file(point_cloud_path)
        else:
            raise ValueError(f"Unsupported point cloud format: {point_cloud_path.suffix}")
        
        # Store point cloud
        self.point_clouds[cloud_id] = {
            'points': points,
            'file_path': point_cloud_path,
            'loaded_at': datetime.now().isoformat()
        }
        
        # Set as current if first loaded
        if self.current_point_cloud is None:
            self.current_point_cloud = cloud_id
        
        logger.info(f"Loaded point cloud '{cloud_id}' with {len(points)} points")
        return cloud_id
    
    def _load_ply_file(self, file_path: Path) -> np.ndarray:
        """Load PLY point cloud file"""
        if O3D_AVAILABLE:
            pcd = o3d.io.read_point_cloud(str(file_path))
            return np.asarray(pcd.points)
        else:
            # Fallback: simple PLY parser
            return self._simple_ply_parser(file_path)
    
    def _load_npy_file(self, file_path: Path) -> np.ndarray:
        """Load NumPy point cloud file"""
        points = np.load(file_path)
        
        # Ensure correct shape (N, 3)
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        elif points.shape[1] != 3:
            raise ValueError(f"Point cloud must have 3 coordinates, got shape: {points.shape}")
        
        return points
    
    def _simple_ply_parser(self, file_path: Path) -> np.ndarray:
        """Simple PLY file parser (fallback when Open3D not available)"""
        points = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find header info
        vertex_count = 0
        data_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('end_header'):
                data_start = i + 1
                break
        
        # Read vertex data
        for i in range(data_start, data_start + vertex_count):
            parts = lines[i].strip().split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                points.append([x, y, z])
        
        return np.array(points)
    
    def select_point_interactive(
        self, 
        cloud_id: Optional[str] = None,
        visualization_backend: str = "open3d"
    ) -> MeasurementPoint:
        """
        Interactive 3D point selection
        
        Args:
            cloud_id: Point cloud to use for selection
            visualization_backend: "open3d" or "matplotlib"
            
        Returns:
            Selected measurement point
        """
        cloud_id = cloud_id or self.current_point_cloud
        
        if cloud_id not in self.point_clouds:
            raise ValueError(f"Point cloud '{cloud_id}' not loaded")
        
        points = self.point_clouds[cloud_id]['points']
        
        if visualization_backend == "open3d" and O3D_AVAILABLE:
            return self._select_point_open3d(points, cloud_id)
        elif visualization_backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            return self._select_point_matplotlib(points, cloud_id)
        else:
            # Fallback: programmatic selection
            logger.warning("Interactive selection not available, using center point")
            center = np.mean(points, axis=0)
            return MeasurementPoint.from_array(center, frame_id=0)
    
    def _select_point_open3d(self, points: np.ndarray, cloud_id: str) -> MeasurementPoint:
        """Open3D-based interactive point selection"""
        logger.info("Starting Open3D interactive point selection...")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color the point cloud
        colors = np.tile([0.7, 0.7, 0.7], (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create visualizer
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Select Point for Measurement")
        vis.add_geometry(pcd)
        
        print("Instructions:")
        print("- Hold Shift and click to select a point")
        print("- Press Q to finish selection")
        
        vis.run()
        selected_indices = vis.get_picked_points()
        vis.destroy_window()
        
        if len(selected_indices) == 0:
            raise ValueError("No point selected")
        
        # Get the selected point
        selected_point = points[selected_indices[0]]
        
        logger.info(f"Selected point: {selected_point}")
        return MeasurementPoint.from_array(selected_point)
    
    def _select_point_matplotlib(self, points: np.ndarray, cloud_id: str) -> MeasurementPoint:
        """Matplotlib-based interactive point selection"""
        logger.info("Starting Matplotlib interactive point selection...")
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot point cloud
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  alpha=0.6, s=1, c='gray')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title('Click to select measurement point')
        
        selected_point = None
        
        def on_click(event):
            nonlocal selected_point
            if event.inaxes == ax:
                # Find closest point to click
                if hasattr(event, 'xdata') and hasattr(event, 'ydata'):
                    # Project 3D points to 2D screen coordinates
                    # This is a simplified approach
                    click_x, click_y = event.xdata, event.ydata
                    
                    # Find closest point (simplified)
                    center = np.mean(points, axis=0)
                    selected_point = center
                    
                    ax.scatter([center[0]], [center[1]], [center[2]], 
                             c='red', s=100, marker='o')
                    plt.draw()
                    
                    print(f"Selected point: {selected_point}")
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        
        if selected_point is None:
            # Default to center
            selected_point = np.mean(points, axis=0)
        
        return MeasurementPoint.from_array(selected_point)
    
    def select_point_programmatic(
        self, 
        coordinates: Tuple[float, float, float],
        cloud_id: Optional[str] = None,
        snap_to_nearest: bool = True
    ) -> MeasurementPoint:
        """
        Programmatically select a point by coordinates
        
        Args:
            coordinates: (x, y, z) coordinates
            cloud_id: Point cloud to use
            snap_to_nearest: Whether to snap to nearest actual point
            
        Returns:
            Measurement point
        """
        cloud_id = cloud_id or self.current_point_cloud
        
        if cloud_id not in self.point_clouds:
            raise ValueError(f"Point cloud '{cloud_id}' not loaded")
        
        target_point = np.array(coordinates)
        
        if snap_to_nearest:
            points = self.point_clouds[cloud_id]['points']
            distances = np.linalg.norm(points - target_point, axis=1)
            nearest_idx = np.argmin(distances)
            actual_point = points[nearest_idx]
            
            logger.info(f"Snapped to nearest point: {actual_point} (distance: {distances[nearest_idx]:.6f})")
            return MeasurementPoint.from_array(actual_point)
        else:
            return MeasurementPoint.from_array(target_point)
    
    def measure_distance(
        self, 
        point_a: MeasurementPoint, 
        point_b: MeasurementPoint,
        validate: bool = True
    ) -> DistanceMeasurement:
        """
        Calculate distance between two measurement points
        
        Args:
            point_a: First measurement point
            point_b: Second measurement point
            validate: Whether to validate the measurement
            
        Returns:
            Distance measurement object
        """
        # Calculate Euclidean distance
        distance = np.linalg.norm(point_a.to_array() - point_b.to_array())
        
        # Create measurement object
        measurement = DistanceMeasurement(
            point_a=point_a,
            point_b=point_b,
            distance=distance,
            unit=self.config['unit']
        )
        
        # Validate measurement if requested
        if validate and self.config['enable_validation']:
            self._validate_measurement(measurement)
        
        # Store measurement
        self.measurements.append(measurement)
        
        logger.info(f"Distance measured: {distance:.{self.config['precision']}f} {self.config['unit']}")
        return measurement
    
    def _validate_measurement(self, measurement: DistanceMeasurement):
        """Validate a distance measurement"""
        distance = measurement.distance
        
        # Check distance thresholds
        if distance < self.config['min_distance_threshold']:
            logger.warning(f"Distance {distance:.6f} is below minimum threshold")
        
        if distance > self.config['max_distance_threshold']:
            logger.warning(f"Distance {distance:.6f} is above maximum threshold")
        
        # Calculate accuracy estimate based on point cloud resolution
        if self.current_point_cloud and self.current_point_cloud in self.point_clouds:
            points = self.point_clouds[self.current_point_cloud]['points']
            
            # Estimate point cloud resolution
            if len(points) > 100:
                sample_indices = np.random.choice(len(points), 100, replace=False)
                sample_points = points[sample_indices]
                distances = []
                
                for i, point in enumerate(sample_points):
                    other_points = np.delete(sample_points, i, axis=0)
                    min_dist = np.min(np.linalg.norm(other_points - point, axis=1))
                    distances.append(min_dist)
                
                avg_resolution = np.mean(distances)
                measurement.accuracy_estimate = avg_resolution * 2  # Rough estimate
    
    def batch_measure_distances(
        self, 
        point_pairs: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
        cloud_id: Optional[str] = None
    ) -> List[DistanceMeasurement]:
        """
        Measure multiple distances in batch
        
        Args:
            point_pairs: List of coordinate pairs to measure
            cloud_id: Point cloud to use
            
        Returns:
            List of distance measurements
        """
        measurements = []
        
        for i, (coords_a, coords_b) in enumerate(point_pairs):
            logger.info(f"Processing measurement {i+1}/{len(point_pairs)}")
            
            point_a = self.select_point_programmatic(coords_a, cloud_id)
            point_b = self.select_point_programmatic(coords_b, cloud_id)
            
            measurement = self.measure_distance(point_a, point_b)
            measurements.append(measurement)
        
        logger.info(f"Completed {len(measurements)} batch measurements")
        return measurements
    
    def export_measurements(
        self, 
        output_path: Union[str, Path],
        format: str = "json"
    ):
        """
        Export measurements to file
        
        Args:
            output_path: Path to output file
            format: Export format ("json", "csv", "txt")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            self._export_json(output_path)
        elif format.lower() == "csv":
            self._export_csv(output_path)
        elif format.lower() == "txt":
            self._export_txt(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(self.measurements)} measurements to {output_path}")
    
    def _export_json(self, output_path: Path):
        """Export measurements to JSON format"""
        export_data = {
            'measurements': [asdict(m) for m in self.measurements],
            'point_clouds': {
                cloud_id: {
                    'file_path': str(data['file_path']),
                    'loaded_at': data['loaded_at'],
                    'point_count': len(data['points'])
                }
                for cloud_id, data in self.point_clouds.items()
            },
            'config': self.config,
            'exported_at': datetime.now().isoformat(),
            'total_measurements': len(self.measurements)
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _export_csv(self, output_path: Path):
        """Export measurements to CSV format"""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'measurement_id', 'distance', 'unit', 'timestamp',
                'point_a_x', 'point_a_y', 'point_a_z',
                'point_b_x', 'point_b_y', 'point_b_z',
                'accuracy_estimate'
            ])
            
            # Data
            for m in self.measurements:
                writer.writerow([
                    m.measurement_id, m.distance, m.unit, m.timestamp,
                    m.point_a.x, m.point_a.y, m.point_a.z,
                    m.point_b.x, m.point_b.y, m.point_b.z,
                    m.accuracy_estimate
                ])
    
    def _export_txt(self, output_path: Path):
        """Export measurements to text format"""
        with open(output_path, 'w') as f:
            f.write("Gauge 3D Distance Measurements\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total measurements: {len(self.measurements)}\n")
            f.write(f"Export time: {datetime.now().isoformat()}\n\n")
            
            for i, m in enumerate(self.measurements, 1):
                f.write(f"Measurement {i}:\n")
                f.write(f"  ID: {m.measurement_id}\n")
                f.write(f"  Distance: {m.distance:.{self.config['precision']}f} {m.unit}\n")
                f.write(f"  Point A: ({m.point_a.x:.6f}, {m.point_a.y:.6f}, {m.point_a.z:.6f})\n")
                f.write(f"  Point B: ({m.point_b.x:.6f}, {m.point_b.y:.6f}, {m.point_b.z:.6f})\n")
                f.write(f"  Timestamp: {m.timestamp}\n")
                if m.accuracy_estimate:
                    f.write(f"  Accuracy estimate: ±{m.accuracy_estimate:.6f} {m.unit}\n")
                f.write("\n")
    
    def visualize_measurements(
        self, 
        cloud_id: Optional[str] = None,
        show_points: bool = True,
        show_lines: bool = True,
        backend: str = "open3d"
    ):
        """
        Visualize measurements on point cloud
        
        Args:
            cloud_id: Point cloud to visualize
            show_points: Whether to show measurement points
            show_lines: Whether to show distance lines
            backend: Visualization backend ("open3d" or "matplotlib")
        """
        cloud_id = cloud_id or self.current_point_cloud
        
        if cloud_id not in self.point_clouds:
            raise ValueError(f"Point cloud '{cloud_id}' not loaded")
        
        if backend == "open3d" and O3D_AVAILABLE:
            self._visualize_open3d(cloud_id, show_points, show_lines)
        elif backend == "matplotlib" and MATPLOTLIB_AVAILABLE:
            self._visualize_matplotlib(cloud_id, show_points, show_lines)
        else:
            logger.warning("Visualization backend not available")
    
    def _visualize_open3d(self, cloud_id: str, show_points: bool, show_lines: bool):
        """Open3D visualization of measurements"""
        points = self.point_clouds[cloud_id]['points']
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile([0.7, 0.7, 0.7], (len(points), 1)))
        
        geometries = [pcd]
        
        # Add measurement visualizations
        for measurement in self.measurements:
            if show_points:
                # Point A
                sphere_a = o3d.geometry.TriangleMesh.create_sphere(radius=self.config['point_size'] * 0.001)
                sphere_a.translate(measurement.point_a.to_array())
                sphere_a.paint_uniform_color(self.config['measurement_color'])
                geometries.append(sphere_a)
                
                # Point B
                sphere_b = o3d.geometry.TriangleMesh.create_sphere(radius=self.config['point_size'] * 0.001)
                sphere_b.translate(measurement.point_b.to_array())
                sphere_b.paint_uniform_color(self.config['measurement_color'])
                geometries.append(sphere_b)
            
            if show_lines:
                # Distance line
                line_points = [measurement.point_a.to_array(), measurement.point_b.to_array()]
                lines = [[0, 1]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(line_points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector([self.config['measurement_color']])
                geometries.append(line_set)
        
        # Visualize
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Distance Measurements - {cloud_id}"
        )
    
    def _visualize_matplotlib(self, cloud_id: str, show_points: bool, show_lines: bool):
        """Matplotlib visualization of measurements"""
        points = self.point_clouds[cloud_id]['points']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot point cloud
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  alpha=0.3, s=1, c='gray')
        
        # Plot measurements
        for i, measurement in enumerate(self.measurements):
            if show_points:
                # Measurement points
                ax.scatter([measurement.point_a.x], [measurement.point_a.y], [measurement.point_a.z],
                         c='red', s=self.config['point_size'] * 10, marker='o')
                ax.scatter([measurement.point_b.x], [measurement.point_b.y], [measurement.point_b.z],
                         c='red', s=self.config['point_size'] * 10, marker='o')
            
            if show_lines:
                # Distance line
                ax.plot([measurement.point_a.x, measurement.point_b.x],
                       [measurement.point_a.y, measurement.point_b.y],
                       [measurement.point_a.z, measurement.point_b.z],
                       color='red', linewidth=self.config['line_width'])
            
            if self.config['show_annotations']:
                # Distance annotation
                mid_x = (measurement.point_a.x + measurement.point_b.x) / 2
                mid_y = (measurement.point_a.y + measurement.point_b.y) / 2
                mid_z = (measurement.point_a.z + measurement.point_b.z) / 2
                
                ax.text(mid_x, mid_y, mid_z, 
                       f'{measurement.distance:.{self.config["precision"]}f}m',
                       fontsize=8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Distance Measurements - {cloud_id}')
        
        plt.show()
    
    def get_measurement_statistics(self) -> Dict:
        """Get statistics about current measurements"""
        if not self.measurements:
            return {"total_measurements": 0}
        
        distances = [m.distance for m in self.measurements]
        
        return {
            "total_measurements": len(self.measurements),
            "min_distance": min(distances),
            "max_distance": max(distances),
            "mean_distance": np.mean(distances),
            "median_distance": np.median(distances),
            "std_distance": np.std(distances),
            "unit": self.config['unit']
        }


def main():
    """Example usage of the distance measurement tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gauge 3D Distance Measurement Tool")
    parser.add_argument("point_cloud", help="Path to point cloud file")
    parser.add_argument("--output", "-o", help="Output file for measurements")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--format", "-f", default="json", choices=["json", "csv", "txt"], help="Output format")
    
    args = parser.parse_args()
    
    # Initialize measurement tool
    tool = DistanceMeasurementTool()
    
    # Load point cloud
    cloud_id = tool.load_point_cloud(args.point_cloud)
    
    if args.interactive:
        print("Interactive measurement mode")
        print("Select two points to measure distance...")
        
        try:
            point_a = tool.select_point_interactive(cloud_id)
            point_b = tool.select_point_interactive(cloud_id)
            
            measurement = tool.measure_distance(point_a, point_b)
            print(f"Distance: {measurement.distance:.3f} {measurement.unit}")
            
        except Exception as e:
            print(f"Interactive selection failed: {e}")
            print("Using example coordinates instead...")
            
            # Example measurement
            point_a = tool.select_point_programmatic((0, 0, 0), cloud_id)
            point_b = tool.select_point_programmatic((1, 1, 1), cloud_id)
            measurement = tool.measure_distance(point_a, point_b)
            print(f"Example distance: {measurement.distance:.3f} {measurement.unit}")
    
    else:
        # Example programmatic measurements
        print("Running example measurements...")
        
        example_pairs = [
            ((0, 0, 0), (1, 0, 0)),
            ((0, 0, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 0, 1))
        ]
        
        measurements = tool.batch_measure_distances(example_pairs, cloud_id)
        
        for i, m in enumerate(measurements, 1):
            print(f"Measurement {i}: {m.distance:.3f} {m.unit}")
    
    # Export results
    if args.output:
        tool.export_measurements(args.output, args.format)
        print(f"Results exported to: {args.output}")
    
    # Show statistics
    stats = tool.get_measurement_statistics()
    print(f"\nMeasurement Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
