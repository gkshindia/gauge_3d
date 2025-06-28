"""
Utility functions for depth estimation pipeline
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def load_depth_map(depth_path: str) -> np.ndarray:
    """
    Load depth map from various formats.
    
    Args:
        depth_path: Path to depth map file
        
    Returns:
        Depth map as numpy array
    """
    depth_path = Path(depth_path)
    
    if depth_path.suffix.lower() == '.npy':
        return np.load(depth_path)
    elif depth_path.suffix.lower() == '.png':
        # Assume 16-bit depth map
        depth_16bit = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_16bit is None:
            raise ValueError(f"Could not load depth map: {depth_path}")
        return depth_16bit.astype(np.float32) / 65535.0
    elif depth_path.suffix.lower() in ['.exr', '.hdr']:
        # Try to load EXR/HDR format
        try:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError("Could not load EXR/HDR file")
            return depth.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Could not load EXR/HDR depth map: {e}")
    else:
        raise ValueError(f"Unsupported depth format: {depth_path.suffix}")


def save_depth_map(depth_map: np.ndarray, output_path: str, format_type: str = "npy") -> str:
    """
    Save depth map in specified format.
    
    Args:
        depth_map: Depth map to save
        output_path: Output file path
        format_type: Output format ("npy", "png", "exr")
        
    Returns:
        Actual output path
    """
    output_path = Path(output_path)
    
    if format_type == "npy":
        output_path = output_path.with_suffix('.npy')
        np.save(output_path, depth_map)
    elif format_type == "png":
        output_path = output_path.with_suffix('.png')
        # Convert to 16-bit PNG
        depth_normalized = np.clip(depth_map, 0, 1)
        depth_16bit = (depth_normalized * 65535).astype(np.uint16)
        cv2.imwrite(str(output_path), depth_16bit)
    elif format_type == "exr":
        output_path = output_path.with_suffix('.exr')
        # Save as EXR (requires OpenCV with EXR support)
        try:
            cv2.imwrite(str(output_path), depth_map.astype(np.float32))
        except Exception as e:
            logger.warning(f"Could not save EXR format: {e}, falling back to NPY")
            output_path = output_path.with_suffix('.npy')
            np.save(output_path, depth_map)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    return str(output_path)


def normalize_depth_map(depth_map: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize depth map using various methods.
    
    Args:
        depth_map: Input depth map
        method: Normalization method ("minmax", "zscore", "percentile")
        
    Returns:
        Normalized depth map
    """
    # Remove invalid values
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    
    if not np.any(valid_mask):
        logger.warning("No valid depth values for normalization")
        return np.zeros_like(depth_map)
    
    if method == "minmax":
        depth_min = np.min(depth_map[valid_mask])
        depth_max = np.max(depth_map[valid_mask])
        if depth_max > depth_min:
            normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth_map)
    elif method == "zscore":
        depth_mean = np.mean(depth_map[valid_mask])
        depth_std = np.std(depth_map[valid_mask])
        if depth_std > 0:
            normalized = (depth_map - depth_mean) / depth_std
            # Clip to reasonable range
            normalized = np.clip(normalized, -3, 3)
            # Scale to [0, 1]
            normalized = (normalized + 3) / 6
        else:
            normalized = np.zeros_like(depth_map)
    elif method == "percentile":
        depth_p05 = np.percentile(depth_map[valid_mask], 5)
        depth_p95 = np.percentile(depth_map[valid_mask], 95)
        if depth_p95 > depth_p05:
            normalized = np.clip((depth_map - depth_p05) / (depth_p95 - depth_p05), 0, 1)
        else:
            normalized = np.zeros_like(depth_map)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Preserve invalid regions
    normalized[~valid_mask] = 0
    
    return normalized


def create_depth_colormap(depth_map: np.ndarray, colormap: str = "plasma") -> np.ndarray:
    """
    Create colored visualization of depth map.
    
    Args:
        depth_map: Input depth map
        colormap: OpenCV colormap name
        
    Returns:
        Color-mapped depth visualization (BGR)
    """
    # Normalize depth map
    normalized = normalize_depth_map(depth_map, "minmax")
    
    # Convert to 8-bit
    depth_8bit = (normalized * 255).astype(np.uint8)
    
    # Apply colormap
    colormap_dict = {
        "plasma": cv2.COLORMAP_PLASMA,
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL
    }
    
    cv_colormap = colormap_dict.get(colormap, cv2.COLORMAP_PLASMA)
    colored = cv2.applyColorMap(depth_8bit, cv_colormap)
    
    return colored


def calculate_depth_statistics(depth_map: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive depth map statistics.
    
    Args:
        depth_map: Input depth map
        
    Returns:
        Dictionary of statistics
    """
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    valid_depths = depth_map[valid_mask]
    
    if len(valid_depths) == 0:
        return {
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'median': 0.0, 'p25': 0.0, 'p75': 0.0, 'range': 0.0,
            'coverage': 0.0, 'valid_pixels': 0, 'total_pixels': depth_map.size
        }
    
    stats = {
        'mean': float(np.mean(valid_depths)),
        'std': float(np.std(valid_depths)),
        'min': float(np.min(valid_depths)),
        'max': float(np.max(valid_depths)),
        'median': float(np.median(valid_depths)),
        'p25': float(np.percentile(valid_depths, 25)),
        'p75': float(np.percentile(valid_depths, 75)),
        'range': float(np.max(valid_depths) - np.min(valid_depths)),
        'coverage': float(np.sum(valid_mask) / depth_map.size),
        'valid_pixels': int(np.sum(valid_mask)),
        'total_pixels': int(depth_map.size)
    }
    
    return stats


def filter_depth_map(depth_map: np.ndarray, method: str = "bilateral", **kwargs) -> np.ndarray:
    """
    Apply filtering to depth map for noise reduction.
    
    Args:
        depth_map: Input depth map
        method: Filtering method ("bilateral", "gaussian", "median")
        **kwargs: Additional parameters for filtering
        
    Returns:
        Filtered depth map
    """
    # Convert to appropriate format for filtering
    depth_float = depth_map.astype(np.float32)
    
    if method == "bilateral":
        d = kwargs.get('d', 5)
        sigma_color = kwargs.get('sigma_color', 50)
        sigma_space = kwargs.get('sigma_space', 50)
        
        filtered = cv2.bilateralFilter(depth_float, d, sigma_color, sigma_space)
    elif method == "gaussian":
        kernel_size = kwargs.get('kernel_size', 5)
        sigma = kwargs.get('sigma', 1.0)
        
        filtered = cv2.GaussianBlur(depth_float, (kernel_size, kernel_size), sigma)
    elif method == "median":
        kernel_size = kwargs.get('kernel_size', 5)
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        filtered = cv2.medianBlur(depth_float, kernel_size)
    else:
        raise ValueError(f"Unknown filtering method: {method}")
    
    return filtered


def resize_depth_map(depth_map: np.ndarray, target_size: Tuple[int, int], 
                    interpolation: str = "linear") -> np.ndarray:
    """
    Resize depth map with appropriate interpolation.
    
    Args:
        depth_map: Input depth map
        target_size: Target size as (width, height)
        interpolation: Interpolation method ("linear", "nearest", "cubic")
        
    Returns:
        Resized depth map
    """
    interp_dict = {
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA
    }
    
    cv_interp = interp_dict.get(interpolation, cv2.INTER_LINEAR)
    resized = cv2.resize(depth_map, target_size, interpolation=cv_interp)
    
    return resized


def detect_depth_edges(depth_map: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Detect edges in depth map for quality assessment.
    
    Args:
        depth_map: Input depth map
        threshold: Edge detection threshold
        
    Returns:
        Binary edge map
    """
    # Calculate gradients
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold to get edges
    edges = gradient_magnitude > threshold
    
    return edges.astype(np.uint8) * 255


def validate_depth_consistency(depth_sequence: List[np.ndarray], 
                             threshold: float = 0.1) -> List[float]:
    """
    Validate temporal consistency of depth sequence.
    
    Args:
        depth_sequence: List of depth maps
        threshold: Consistency threshold
        
    Returns:
        List of consistency scores for each frame pair
    """
    if len(depth_sequence) < 2:
        return []
    
    consistency_scores = []
    
    for i in range(1, len(depth_sequence)):
        prev_depth = depth_sequence[i-1]
        curr_depth = depth_sequence[i]
        
        if prev_depth.shape != curr_depth.shape:
            # Resize to match
            curr_depth = resize_depth_map(curr_depth, prev_depth.shape[1::-1])
        
        # Calculate absolute difference
        diff = np.abs(curr_depth - prev_depth)
        
        # Get valid regions
        valid_mask = (np.isfinite(prev_depth) & np.isfinite(curr_depth) & 
                     (prev_depth > 0) & (curr_depth > 0))
        
        if np.any(valid_mask):
            mean_diff = np.mean(diff[valid_mask])
            depth_range = np.max([np.max(prev_depth[valid_mask]), 
                                np.max(curr_depth[valid_mask])]) - \
                         np.min([np.min(prev_depth[valid_mask]), 
                                np.min(curr_depth[valid_mask])])
            
            # Normalize by depth range
            if depth_range > 0:
                consistency = 1.0 - (mean_diff / depth_range)
            else:
                consistency = 1.0
        else:
            consistency = 0.0
        
        consistency_scores.append(max(0.0, consistency))
    
    return consistency_scores


def create_side_by_side_comparison(image: np.ndarray, depth_map: np.ndarray, 
                                 title: Optional[str] = None) -> np.ndarray:
    """
    Create side-by-side comparison of image and depth map.
    
    Args:
        image: Original image (BGR)
        depth_map: Depth map
        title: Optional title
        
    Returns:
        Combined visualization
    """
    # Ensure same height
    if image.shape[0] != depth_map.shape[0] or image.shape[1] != depth_map.shape[1]:
        depth_map = resize_depth_map(depth_map, (image.shape[1], image.shape[0]))
    
    # Create colored depth map
    depth_colored = create_depth_colormap(depth_map, "plasma")
    
    # Combine side by side
    combined = np.hstack([image, depth_colored])
    
    # Add title if provided
    if title:
        # Add text space at top
        text_height = 40
        text_area = np.zeros((text_height, combined.shape[1], 3), dtype=np.uint8)
        
        # Add text
        cv2.putText(text_area, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        combined = np.vstack([text_area, combined])
    
    return combined


def get_depth_map_info(depth_path: str) -> Dict[str, Any]:
    """
    Get information about a depth map file.
    
    Args:
        depth_path: Path to depth map file
        
    Returns:
        Information dictionary
    """
    depth_path = Path(depth_path)
    
    info = {
        'path': str(depth_path),
        'filename': depth_path.name,
        'format': depth_path.suffix,
        'exists': depth_path.exists(),
        'size_bytes': depth_path.stat().st_size if depth_path.exists() else 0
    }
    
    if depth_path.exists():
        try:
            depth_map = load_depth_map(str(depth_path))
            info.update({
                'shape': depth_map.shape,
                'dtype': str(depth_map.dtype),
                'statistics': calculate_depth_statistics(depth_map)
            })
        except Exception as e:
            info['load_error'] = str(e)
    
    return info
