"""
Depth Map Output Management and Validation
Step 1.4: Output Management for depth estimation pipeline
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass, asdict
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OutputConfig:
    """Configuration for output management."""
    # Directory structure
    base_output_dir: str = "output"
    organize_by_date: bool = True
    create_video_subdirs: bool = True
    
    # File formats
    depth_format: str = "npy"  # "npy", "exr", "png"
    preview_format: str = "jpg"  # "jpg", "png"
    metadata_format: str = "json"  # "json", "yaml"
    
    # Quality metrics
    enable_quality_validation: bool = True
    depth_consistency_threshold: float = 0.1
    gradient_threshold: float = 50.0
    
    # Archive and cleanup
    enable_archiving: bool = True
    keep_intermediate_files: bool = False
    compress_archives: bool = True


@dataclass
class DepthQualityMetrics:
    """Quality metrics for depth maps."""
    # Basic statistics
    mean_depth: float
    std_depth: float
    min_depth: float
    max_depth: float
    depth_range: float
    
    # Quality indicators
    gradient_magnitude: float
    smoothness_score: float
    consistency_score: float
    coverage_ratio: float  # Ratio of valid depth values
    
    # Temporal metrics (for sequences)
    temporal_consistency: Optional[float] = None
    motion_coherence: Optional[float] = None
    
    # Overall quality score
    overall_quality: float = 0.0


class DepthMapValidator:
    """Validate depth map quality and consistency."""
    
    def __init__(self, config: OutputConfig):
        self.config = config
    
    def validate_depth_map(self, depth_map: np.ndarray, 
                          prev_depth: Optional[np.ndarray] = None) -> DepthQualityMetrics:
        """
        Validate a single depth map and calculate quality metrics.
        
        Args:
            depth_map: Depth map to validate
            prev_depth: Previous depth map for temporal consistency
            
        Returns:
            Quality metrics
        """
        # Basic statistics
        valid_mask = np.isfinite(depth_map) & (depth_map > 0)
        valid_depths = depth_map[valid_mask]
        
        if len(valid_depths) == 0:
            logger.warning("No valid depth values found")
            return self._create_empty_metrics()
        
        mean_depth = float(np.mean(valid_depths))
        std_depth = float(np.std(valid_depths))
        min_depth = float(np.min(valid_depths))
        max_depth = float(np.max(valid_depths))
        depth_range = max_depth - min_depth
        coverage_ratio = float(np.sum(valid_mask) / depth_map.size)
        
        # Calculate gradients
        gradient_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = float(np.mean(np.sqrt(gradient_x**2 + gradient_y**2)))
        
        # Smoothness score (inverse of gradient variance)
        gradient_var = np.var(np.sqrt(gradient_x**2 + gradient_y**2))
        smoothness_score = float(1.0 / (1.0 + gradient_var))
        
        # Consistency score (based on local smoothness)
        laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)
        consistency_score = float(1.0 / (1.0 + np.var(laplacian)))
        
        # Temporal consistency (if previous depth available)
        temporal_consistency = None
        if prev_depth is not None and prev_depth.shape == depth_map.shape:
            diff = np.abs(depth_map - prev_depth)
            temporal_consistency = float(1.0 - np.mean(diff) / (depth_range + 1e-8))
        
        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(
            smoothness_score, consistency_score, coverage_ratio, temporal_consistency
        )
        
        return DepthQualityMetrics(
            mean_depth=mean_depth,
            std_depth=std_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            depth_range=depth_range,
            gradient_magnitude=gradient_magnitude,
            smoothness_score=smoothness_score,
            consistency_score=consistency_score,
            coverage_ratio=coverage_ratio,
            temporal_consistency=temporal_consistency,
            overall_quality=overall_quality
        )
    
    def _create_empty_metrics(self) -> DepthQualityMetrics:
        """Create empty metrics for invalid depth maps."""
        return DepthQualityMetrics(
            mean_depth=0.0,
            std_depth=0.0,
            min_depth=0.0,
            max_depth=0.0,
            depth_range=0.0,
            gradient_magnitude=0.0,
            smoothness_score=0.0,
            consistency_score=0.0,
            coverage_ratio=0.0,
            overall_quality=0.0
        )
    
    def _calculate_overall_quality(self, smoothness: float, consistency: float, 
                                 coverage: float, temporal: Optional[float]) -> float:
        """Calculate overall quality score."""
        # Weighted combination of quality factors
        weights = {
            'smoothness': 0.25,
            'consistency': 0.25,
            'coverage': 0.25,
            'temporal': 0.25
        }
        
        score = (weights['smoothness'] * smoothness + 
                weights['consistency'] * consistency + 
                weights['coverage'] * coverage)
        
        if temporal is not None:
            score += weights['temporal'] * temporal
        else:
            # Redistribute temporal weight to other factors
            score = score / (1.0 - weights['temporal'])
        
        return float(np.clip(score, 0.0, 1.0))


class DepthMapVisualizer:
    """Create visualizations and previews of depth maps."""
    
    def __init__(self, config: OutputConfig):
        self.config = config
    
    def create_depth_preview(self, frame: np.ndarray, depth_map: np.ndarray,
                           output_path: str, title: Optional[str] = None) -> str:
        """
        Create a comprehensive depth map preview.
        
        Args:
            frame: Original frame
            depth_map: Depth map
            output_path: Output file path
            title: Optional title for the preview
            
        Returns:
            Path to created preview
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        # Original frame
        axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Frame")
        axes[0, 0].axis('off')
        
        # Depth map (plasma colormap)
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        im1 = axes[0, 1].imshow(depth_normalized, cmap='plasma')
        axes[0, 1].set_title("Depth Map (Plasma)")
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Depth map (jet colormap)
        im2 = axes[1, 0].imshow(depth_normalized, cmap='jet')
        axes[1, 0].set_title("Depth Map (Jet)")
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Depth histogram
        valid_depths = depth_map[np.isfinite(depth_map) & (depth_map > 0)]
        if len(valid_depths) > 0:
            axes[1, 1].hist(valid_depths, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title("Depth Distribution")
            axes[1, 1].set_xlabel("Depth Value")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "No valid depth data", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Depth Distribution")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_quality_report_visualization(self, metrics_list: List[DepthQualityMetrics],
                                          output_path: str, video_name: str) -> str:
        """
        Create quality report visualization for a video sequence.
        
        Args:
            metrics_list: List of quality metrics for each frame
            output_path: Output file path
            video_name: Video name for title
            
        Returns:
            Path to created visualization
        """
        if not metrics_list:
            logger.warning("No metrics to visualize")
            return output_path
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f"Quality Report: {video_name}", fontsize=16)
        
        frame_indices = list(range(len(metrics_list)))
        
        # Overall quality over time
        overall_quality = [m.overall_quality for m in metrics_list]
        axes[0, 0].plot(frame_indices, overall_quality, 'b-', linewidth=2)
        axes[0, 0].set_title("Overall Quality Score")
        axes[0, 0].set_xlabel("Frame Index")
        axes[0, 0].set_ylabel("Quality Score")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Depth statistics
        mean_depths = [m.mean_depth for m in metrics_list]
        std_depths = [m.std_depth for m in metrics_list]
        axes[0, 1].plot(frame_indices, mean_depths, 'g-', label='Mean', linewidth=2)
        axes[0, 1].fill_between(frame_indices, 
                               np.array(mean_depths) - np.array(std_depths),
                               np.array(mean_depths) + np.array(std_depths),
                               alpha=0.3, color='green')
        axes[0, 1].set_title("Depth Statistics")
        axes[0, 1].set_xlabel("Frame Index")
        axes[0, 1].set_ylabel("Depth Value")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Quality components
        smoothness = [m.smoothness_score for m in metrics_list]
        consistency = [m.consistency_score for m in metrics_list]
        coverage = [m.coverage_ratio for m in metrics_list]
        
        axes[1, 0].plot(frame_indices, smoothness, 'r-', label='Smoothness', linewidth=2)
        axes[1, 0].plot(frame_indices, consistency, 'b-', label='Consistency', linewidth=2)
        axes[1, 0].plot(frame_indices, coverage, 'g-', label='Coverage', linewidth=2)
        axes[1, 0].set_title("Quality Components")
        axes[1, 0].set_xlabel("Frame Index")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Temporal consistency (if available)
        temporal_consistency = [m.temporal_consistency for m in metrics_list 
                              if m.temporal_consistency is not None]
        if temporal_consistency:
            temporal_indices = frame_indices[1:len(temporal_consistency)+1]
            axes[1, 1].plot(temporal_indices, temporal_consistency, 'm-', linewidth=2)
            axes[1, 1].set_title("Temporal Consistency")
            axes[1, 1].set_xlabel("Frame Index")
            axes[1, 1].set_ylabel("Consistency Score")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
        else:
            axes[1, 1].text(0.5, 0.5, "No temporal data", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Temporal Consistency")
        
        # Depth range over time
        depth_ranges = [m.depth_range for m in metrics_list]
        axes[2, 0].plot(frame_indices, depth_ranges, 'c-', linewidth=2)
        axes[2, 0].set_title("Depth Range")
        axes[2, 0].set_xlabel("Frame Index")
        axes[2, 0].set_ylabel("Range")
        axes[2, 0].grid(True, alpha=0.3)
        
        # Quality distribution histogram
        axes[2, 1].hist(overall_quality, bins=20, alpha=0.7, edgecolor='black')
        axes[2, 1].set_title("Quality Score Distribution")
        axes[2, 1].set_xlabel("Quality Score")
        axes[2, 1].set_ylabel("Frequency")
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path


class DepthOutputManager:
    """
    Manage depth estimation outputs, validation, and organization.
    Step 1.4: Complete output management for depth estimation pipeline.
    """
    
    def __init__(self, config: OutputConfig):
        """
        Initialize the output manager.
        
        Args:
            config: Output management configuration
        """
        self.config = config
        self.base_dir = Path(config.base_output_dir)
        
        # Initialize components
        self.validator = DepthMapValidator(config)
        self.visualizer = DepthMapVisualizer(config)
        
        # Create directory structure
        self._setup_directory_structure()
        
        logger.info("Initialized DepthOutputManager")
    
    def _setup_directory_structure(self):
        """Set up the output directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.organize_by_date:
            self.session_dir = self.base_dir / f"session_{timestamp}"
        else:
            self.session_dir = self.base_dir
        
        # Create subdirectories
        self.directories = {
            'depth_maps': self.session_dir / "depth_maps",
            'previews': self.session_dir / "previews",
            'quality_reports': self.session_dir / "quality_reports",
            'visualizations': self.session_dir / "visualizations",
            'metadata': self.session_dir / "metadata",
            'archives': self.session_dir / "archives"
        }
        
        for dir_path in self.directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory structure: {self.session_dir}")
    
    def process_video_output(self, video_name: str, frame_data: List[Dict],
                           original_frames: List[str]) -> Dict[str, Any]:
        """
        Process and organize all outputs for a video.
        
        Args:
            video_name: Name of the video
            frame_data: List of frame processing data
            original_frames: List of original frame paths
            
        Returns:
            Summary of processing results
        """
        logger.info(f"Processing outputs for video: {video_name}")
        
        # Create video-specific directories
        video_dirs = {}
        if self.config.create_video_subdirs:
            for key, base_dir in self.directories.items():
                video_dir = base_dir / video_name
                video_dir.mkdir(parents=True, exist_ok=True)
                video_dirs[key] = video_dir
        else:
            video_dirs = self.directories
        
        # Process each frame
        quality_metrics = []
        organized_files = {
            'depth_maps': [],
            'previews': [],
            'metadata_files': []
        }
        
        prev_depth = None
        
        for i, frame_info in enumerate(frame_data):
            try:
                # Load depth map
                depth_map = self._load_depth_map(frame_info['depth_path'])
                
                # Validate depth map
                metrics = self.validator.validate_depth_map(depth_map, prev_depth)
                quality_metrics.append(metrics)
                
                # Organize depth map file
                organized_depth_path = self._organize_depth_file(
                    frame_info['depth_path'], video_dirs['depth_maps'], i
                )
                organized_files['depth_maps'].append(organized_depth_path)
                
                # Create enhanced preview
                if i < len(original_frames):
                    original_frame = cv2.imread(original_frames[i])
                    if original_frame is not None:
                        preview_path = video_dirs['previews'] / f"frame_{i:06d}_preview.jpg"
                        self.visualizer.create_depth_preview(
                            original_frame, depth_map, str(preview_path),
                            title=f"Frame {i} - Quality: {metrics.overall_quality:.3f}"
                        )
                        organized_files['previews'].append(str(preview_path))
                
                # Save frame metadata
                metadata_path = video_dirs['metadata'] / f"frame_{i:06d}_metadata.json"
                self._save_frame_metadata(frame_info, metrics, str(metadata_path))
                organized_files['metadata_files'].append(str(metadata_path))
                
                prev_depth = depth_map
                
            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                continue
        
        # Create quality report visualization
        quality_viz_path = video_dirs['visualizations'] / f"{video_name}_quality_report.png"
        self.visualizer.create_quality_report_visualization(
            quality_metrics, str(quality_viz_path), video_name
        )
        
        # Create video summary
        summary = self._create_video_summary(video_name, quality_metrics, organized_files)
        
        # Save video summary
        summary_path = video_dirs['metadata'] / f"{video_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Archive if requested
        if self.config.enable_archiving:
            archive_path = self._create_archive(video_name, video_dirs)
            summary['archive_path'] = str(archive_path)
        
        # Cleanup intermediate files
        if not self.config.keep_intermediate_files:
            self._cleanup_intermediate_files(frame_data)
        
        logger.info(f"Completed output processing for {video_name}")
        return summary
    
    def _load_depth_map(self, depth_path: str) -> np.ndarray:
        """Load depth map from file."""
        depth_path = Path(depth_path)
        
        if depth_path.suffix == '.npy':
            return np.load(depth_path)
        elif depth_path.suffix == '.png':
            # Load 16-bit PNG and convert back to float
            depth_16bit = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            return depth_16bit.astype(np.float32) / 65535.0
        else:
            raise ValueError(f"Unsupported depth format: {depth_path.suffix}")
    
    def _organize_depth_file(self, source_path: str, target_dir: Path, frame_index: int) -> str:
        """Organize depth file with proper naming."""
        source_path = Path(source_path)
        target_path = target_dir / f"depth_{frame_index:06d}{source_path.suffix}"
        
        if source_path != target_path:
            shutil.copy2(source_path, target_path)
        
        return str(target_path)
    
    def _save_frame_metadata(self, frame_info: Dict, metrics: DepthQualityMetrics, 
                           metadata_path: str):
        """Save frame metadata including quality metrics."""
        metadata = {
            'frame_info': frame_info,
            'quality_metrics': asdict(metrics),
            'timestamp': datetime.now().isoformat(),
            'validation_passed': metrics.overall_quality >= 0.5  # Quality threshold
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _create_video_summary(self, video_name: str, quality_metrics: List[DepthQualityMetrics],
                            organized_files: Dict) -> Dict[str, Any]:
        """Create comprehensive video processing summary."""
        if not quality_metrics:
            return {
                'video_name': video_name,
                'total_frames': 0,
                'processed_frames': 0,
                'quality_stats': {},
                'files': organized_files
            }
        
        # Calculate aggregate statistics
        overall_qualities = [m.overall_quality for m in quality_metrics]
        depth_ranges = [m.depth_range for m in quality_metrics]
        
        summary = {
            'video_name': video_name,
            'total_frames': len(quality_metrics),
            'processed_frames': len([m for m in quality_metrics if m.overall_quality > 0]),
            'processing_timestamp': datetime.now().isoformat(),
            'quality_stats': {
                'mean_quality': float(np.mean(overall_qualities)),
                'std_quality': float(np.std(overall_qualities)),
                'min_quality': float(np.min(overall_qualities)),
                'max_quality': float(np.max(overall_qualities)),
                'frames_above_threshold': len([q for q in overall_qualities if q >= 0.5]),
                'mean_depth_range': float(np.mean(depth_ranges)),
                'temporal_consistency_available': any(m.temporal_consistency is not None 
                                                    for m in quality_metrics)
            },
            'files': organized_files,
            'config': asdict(self.config)
        }
        
        return summary
    
    def _create_archive(self, video_name: str, video_dirs: Dict[str, Path]) -> str:
        """Create archive of video processing results."""
        import tarfile
        
        archive_path = self.directories['archives'] / f"{video_name}_results.tar"
        
        if self.config.compress_archives:
            archive_path = archive_path.with_suffix('.tar.gz')
            mode = 'w:gz'
        else:
            mode = 'w'
        
        with tarfile.open(archive_path, mode) as tar:
            for dir_type, dir_path in video_dirs.items():
                if dir_path.exists():
                    tar.add(dir_path, arcname=f"{video_name}/{dir_type}")
        
        logger.info(f"Created archive: {archive_path}")
        return str(archive_path)
    
    def _cleanup_intermediate_files(self, frame_data: List[Dict]):
        """Clean up intermediate files if not needed."""
        for frame_info in frame_data:
            # Remove temporary depth files if they were copied
            temp_files = [
                frame_info.get('temp_depth_path'),
                frame_info.get('temp_preview_path')
            ]
            
            for temp_file in temp_files:
                if temp_file and Path(temp_file).exists():
                    try:
                        Path(temp_file).unlink()
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_file}: {e}")


def create_default_output_config() -> OutputConfig:
    """Create default output management configuration."""
    return OutputConfig(
        base_output_dir="output",
        organize_by_date=True,
        create_video_subdirs=True,
        depth_format="npy",
        preview_format="jpg",
        metadata_format="json",
        enable_quality_validation=True,
        depth_consistency_threshold=0.1,
        gradient_threshold=50.0,
        enable_archiving=True,
        keep_intermediate_files=False,
        compress_archives=True
    )


if __name__ == "__main__":
    """Example usage for testing."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="Depth map output management")
    parser.add_argument("depth_dir", help="Directory containing depth maps")
    parser.add_argument("frames_dir", help="Directory containing original frames")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--video-name", default="test_video", help="Video name")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_output_config()
    config.base_output_dir = args.output_dir
    
    # Get depth and frame files
    depth_files = sorted(glob.glob(str(Path(args.depth_dir) / "*.npy")))
    frame_files = sorted(glob.glob(str(Path(args.frames_dir) / "*.jpg")))
    
    if not depth_files:
        print(f"No depth files found in {args.depth_dir}")
        sys.exit(1)
    
    # Create mock frame data
    frame_data = []
    for i, depth_path in enumerate(depth_files):
        frame_data.append({
            'frame_index': i,
            'frame_path': frame_files[i] if i < len(frame_files) else None,
            'depth_path': depth_path,
            'preview_path': None
        })
    
    # Process outputs
    output_manager = DepthOutputManager(config)
    summary = output_manager.process_video_output(args.video_name, frame_data, frame_files)
    
    print("\nOutput Management Results:")
    print(f"Video: {summary['video_name']}")
    print(f"Processed frames: {summary['processed_frames']}")
    print(f"Mean quality: {summary['quality_stats']['mean_quality']:.3f}")
    print(f"Frames above threshold: {summary['quality_stats']['frames_above_threshold']}")
    print(f"Session directory: {output_manager.session_dir}")
