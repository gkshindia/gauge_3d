"""
Utility functions for the transformer enhancement pipeline
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except ImportError:
        logger.error("PyYAML not available. Cannot load YAML configuration.")
        return {}
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def save_config(config: Dict, config_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        import yaml
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to: {config_path}")
        
    except ImportError:
        logger.error("PyYAML not available. Cannot save YAML configuration.")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")


def normalize_point_cloud(points: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Normalize point cloud to unit cube.
    
    Args:
        points: Point cloud array (N, 3)
        
    Returns:
        Normalized points and normalization parameters
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Center and scale
    center = (min_coords + max_coords) / 2
    scale = np.max(max_coords - min_coords)
    
    if scale == 0:
        scale = 1.0
    
    normalized_points = (points - center) / scale
    
    normalization_params = {
        'center': center,
        'scale': scale,
        'min_coords': min_coords,
        'max_coords': max_coords
    }
    
    return normalized_points, normalization_params


def denormalize_point_cloud(points: np.ndarray, params: Dict) -> np.ndarray:
    """
    Denormalize point cloud from unit cube.
    
    Args:
        points: Normalized point cloud array (N, 3)
        params: Normalization parameters
        
    Returns:
        Denormalized points
    """
    center = params['center']
    scale = params['scale']
    
    denormalized_points = points * scale + center
    
    return denormalized_points


def calculate_point_cloud_metrics(points: np.ndarray) -> Dict:
    """
    Calculate basic metrics for a point cloud.
    
    Args:
        points: Point cloud array (N, 3)
        
    Returns:
        Dictionary of metrics
    """
    if len(points) == 0:
        return {
            'num_points': 0,
            'density': 0.0,
            'bounding_box_volume': 0.0,
            'center': np.array([0, 0, 0]),
            'extent': np.array([0, 0, 0])
        }
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = np.mean(points, axis=0)
    extent = max_coords - min_coords
    
    # Calculate bounding box volume
    volume = np.prod(extent)
    
    # Calculate density (points per unit volume)
    density = len(points) / volume if volume > 0 else 0
    
    metrics = {
        'num_points': len(points),
        'density': density,
        'bounding_box_volume': volume,
        'center': center,
        'extent': extent,
        'min_coords': min_coords,
        'max_coords': max_coords
    }
    
    return metrics


def estimate_point_cloud_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Estimate normals for point cloud using PCA.
    
    Args:
        points: Point cloud array (N, 3)
        k: Number of neighbors for normal estimation
        
    Returns:
        Normal vectors array (N, 3)
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        if len(points) < k:
            k = len(points) - 1
        
        if k < 3:
            # Not enough points for meaningful normal estimation
            return np.zeros_like(points)
        
        # Find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        _, indices = nbrs.kneighbors(points)
        
        normals = np.zeros_like(points)
        
        for i, neighbors in enumerate(indices):
            neighbor_points = points[neighbors]
            
            # Center the neighborhood
            centered = neighbor_points - np.mean(neighbor_points, axis=0)
            
            # Compute covariance matrix
            cov_matrix = np.cov(centered.T)
            
            # Find eigenvector with smallest eigenvalue (normal direction)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normal = eigenvectors[:, np.argmin(eigenvalues)]
            
            normals[i] = normal
        
        return normals
        
    except ImportError:
        logger.warning("scikit-learn not available. Cannot estimate normals.")
        return np.zeros_like(points)


def filter_points_by_distance(
    points: np.ndarray, 
    reference_point: np.ndarray, 
    max_distance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points by distance from a reference point.
    
    Args:
        points: Point cloud array (N, 3)
        reference_point: Reference point (3,)
        max_distance: Maximum distance threshold
        
    Returns:
        Filtered points and indices of kept points
    """
    distances = np.linalg.norm(points - reference_point, axis=1)
    mask = distances <= max_distance
    
    filtered_points = points[mask]
    indices = np.where(mask)[0]
    
    return filtered_points, indices


def subsample_point_cloud(
    points: np.ndarray, 
    target_count: int, 
    method: str = "random"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample point cloud to target number of points.
    
    Args:
        points: Point cloud array (N, 3)
        target_count: Target number of points
        method: Subsampling method ("random", "uniform", "fps")
        
    Returns:
        Subsampled points and indices of selected points
    """
    if len(points) <= target_count:
        return points, np.arange(len(points))
    
    if method == "random":
        indices = np.random.choice(len(points), target_count, replace=False)
    elif method == "uniform":
        indices = np.linspace(0, len(points) - 1, target_count, dtype=int)
    elif method == "fps":
        indices = farthest_point_sampling(points, target_count)
    else:
        logger.warning(f"Unknown subsampling method: {method}. Using random.")
        indices = np.random.choice(len(points), target_count, replace=False)
    
    subsampled_points = points[indices]
    return subsampled_points, indices


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest point sampling for point clouds.
    
    Args:
        points: Point cloud array (N, 3)
        num_samples: Number of points to sample
        
    Returns:
        Indices of sampled points
    """
    N = len(points)
    if num_samples >= N:
        return np.arange(N)
    
    # Initialize with random point
    selected_indices = [np.random.randint(0, N)]
    distances = np.full(N, np.inf)
    
    for _ in range(1, num_samples):
        # Update distances to nearest selected point
        last_selected = points[selected_indices[-1]]
        new_distances = np.linalg.norm(points - last_selected, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select farthest point
        farthest_idx = np.argmax(distances)
        selected_indices.append(farthest_idx)
    
    return np.array(selected_indices)


def create_point_cloud_grid(
    points: np.ndarray, 
    voxel_size: float
) -> Tuple[np.ndarray, Dict]:
    """
    Create voxel grid representation of point cloud.
    
    Args:
        points: Point cloud array (N, 3)
        voxel_size: Size of each voxel
        
    Returns:
        Voxel grid and grid information
    """
    if len(points) == 0:
        return np.array([]), {}
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # Calculate grid dimensions
    grid_size = ((max_coords - min_coords) / voxel_size).astype(int) + 1
    
    # Map points to voxel indices
    voxel_indices = ((points - min_coords) / voxel_size).astype(int)
    
    # Create voxel grid
    voxel_grid = np.zeros(grid_size, dtype=bool)
    
    # Mark occupied voxels
    for idx in voxel_indices:
        if np.all(idx >= 0) and np.all(idx < grid_size):
            voxel_grid[tuple(idx)] = True
    
    grid_info = {
        'voxel_size': voxel_size,
        'grid_size': grid_size,
        'min_coords': min_coords,
        'max_coords': max_coords,
        'num_occupied_voxels': np.sum(voxel_grid)
    }
    
    return voxel_grid, grid_info


def save_point_cloud_as_ply(
    points: np.ndarray, 
    colors: Optional[np.ndarray], 
    file_path: Union[str, Path]
):
    """
    Save point cloud as PLY file.
    
    Args:
        points: Point cloud array (N, 3)
        colors: Color array (N, 3), optional
        file_path: Path to save PLY file
    """
    try:
        import open3d as o3d
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(str(file_path), pcd)
        logger.info(f"Saved point cloud to: {file_path}")
        
    except ImportError:
        logger.error("Open3D not available. Cannot save PLY file.")
    except Exception as e:
        logger.error(f"Error saving PLY file: {e}")


def load_point_cloud_from_ply(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud from PLY file.
    
    Args:
        file_path: Path to PLY file
        
    Returns:
        Points and colors arrays
    """
    try:
        import open3d as o3d
        
        pcd = o3d.io.read_point_cloud(str(file_path))
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.array([])
        
        logger.info(f"Loaded point cloud from: {file_path}")
        return points, colors
        
    except ImportError:
        logger.error("Open3D not available. Cannot load PLY file.")
        return np.array([]), np.array([])
    except Exception as e:
        logger.error(f"Error loading PLY file: {e}")
        return np.array([]), np.array([])


def compute_chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    Compute Chamfer distance between two point clouds.
    
    Args:
        points1: First point cloud (N, 3)
        points2: Second point cloud (M, 3)
        
    Returns:
        Chamfer distance
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest neighbors from points1 to points2
        nbrs1 = NearestNeighbors(n_neighbors=1).fit(points2)
        distances1, _ = nbrs1.kneighbors(points1)
        
        # Find nearest neighbors from points2 to points1
        nbrs2 = NearestNeighbors(n_neighbors=1).fit(points1)
        distances2, _ = nbrs2.kneighbors(points2)
        
        # Chamfer distance is sum of both directions
        chamfer_distance = np.mean(distances1) + np.mean(distances2)
        
        return chamfer_distance
        
    except ImportError:
        logger.error("scikit-learn not available. Cannot compute Chamfer distance.")
        return float('inf')


def create_temporal_point_cloud_sequence(
    point_clouds: List[np.ndarray],
    interpolation_factor: int = 1
) -> List[np.ndarray]:
    """
    Create temporally interpolated point cloud sequence.
    
    Args:
        point_clouds: List of point cloud arrays
        interpolation_factor: Number of interpolated frames between each pair
        
    Returns:
        Interpolated point cloud sequence
    """
    if interpolation_factor <= 1 or len(point_clouds) < 2:
        return point_clouds
    
    interpolated_sequence = []
    
    for i in range(len(point_clouds) - 1):
        current_cloud = point_clouds[i]
        next_cloud = point_clouds[i + 1]
        
        # Add current frame
        interpolated_sequence.append(current_cloud)
        
        # Add interpolated frames
        for j in range(1, interpolation_factor):
            t = j / interpolation_factor
            
            # Simple linear interpolation (assumes same number of points)
            if len(current_cloud) == len(next_cloud):
                interpolated_cloud = (1 - t) * current_cloud + t * next_cloud
                interpolated_sequence.append(interpolated_cloud)
    
    # Add last frame
    interpolated_sequence.append(point_clouds[-1])
    
    return interpolated_sequence
