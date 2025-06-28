"""
Main Depth Estimation Pipeline Integration
Combines all Phase 1 components for complete video-to-depth processing
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import Phase 1 components
from setup_environment import DepthAnyVideoSetup as DepthEnvironmentSetup
from video_depth_processor import VideoDepthProcessor, ProcessingConfig, create_default_config
from depth_pipeline import DepthEstimationPipeline, DepthConfig, create_default_depth_config
from output_manager import DepthOutputManager, OutputConfig, create_default_output_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DepthEstimationMaster:
    """
    Master controller for the complete depth estimation pipeline.
    Orchestrates all Phase 1 components for video-to-depth processing.
    """
    
    def __init__(self, 
                 processing_config: Optional[ProcessingConfig] = None,
                 depth_config: Optional[DepthConfig] = None,
                 output_config: Optional[OutputConfig] = None,
                 output_dir: str = "output"):
        """
        Initialize the master depth estimation pipeline.
        
        Args:
            processing_config: Video processing configuration
            depth_config: Depth estimation configuration
            output_config: Output management configuration
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use default configs if not provided
        self.processing_config = processing_config or create_default_config()
        self.depth_config = depth_config or create_default_depth_config()
        self.output_config = output_config or create_default_output_config()
        
        # Update output directory in configs
        self.output_config.base_output_dir = str(self.output_dir)
        
        # Initialize components
        self.video_processor = VideoDepthProcessor(self.processing_config, str(self.output_dir))
        self.depth_pipeline = DepthEstimationPipeline(self.depth_config, str(self.output_dir))
        self.output_manager = DepthOutputManager(self.output_config)
        
        logger.info("Initialized DepthEstimationMaster")
        logger.info(f"Output directory: {self.output_dir}")
    
    def process_video(self, video_path: str, video_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a complete video through the depth estimation pipeline.
        
        Args:
            video_path: Path to input video file
            video_name: Optional name for the video (uses filename if not provided)
            
        Returns:
            Complete processing results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if video_name is None:
            video_name = video_path.stem
        
        logger.info(f"Starting complete depth estimation pipeline for: {video_name}")
        
        results = {
            'video_name': video_name,
            'video_path': str(video_path),
            'pipeline_stages': {},
            'overall_success': False,
            'error_message': None
        }
        
        try:
            # Stage 1: Video Preprocessing
            logger.info("Stage 1: Video preprocessing and frame extraction")
            preprocessing_results = self.video_processor.preprocess_video(str(video_path))
            results['pipeline_stages']['preprocessing'] = preprocessing_results
            
            # Get processed frames
            processed_frames = [f for f in preprocessing_results['processed_frames'] if f['accepted']]
            
            if not processed_frames:
                raise ValueError("No acceptable frames found after preprocessing")
            
            frame_paths = [f['path'] for f in processed_frames]
            logger.info(f"Preprocessed {len(frame_paths)} frames")
            
            # Stage 2: Depth Estimation
            logger.info("Stage 2: Depth estimation with DA-V")
            depth_results = self.depth_pipeline.process_video_sequence(frame_paths, video_name)
            results['pipeline_stages']['depth_estimation'] = depth_results
            
            # Stage 3: Output Management
            logger.info("Stage 3: Output organization and validation")
            output_results = self.output_manager.process_video_output(
                video_name, 
                depth_results['processed_frames'],
                frame_paths
            )
            results['pipeline_stages']['output_management'] = output_results
            
            # Create final summary
            results.update(self._create_final_summary(
                preprocessing_results, depth_results, output_results
            ))
            
            results['overall_success'] = True
            logger.info(f"Successfully completed depth estimation pipeline for {video_name}")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            results['error_message'] = error_msg
            results['overall_success'] = False
        
        # Save complete results
        self._save_pipeline_results(video_name, results)
        
        return results
    
    def process_multiple_videos(self, video_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple videos through the pipeline.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            Batch processing results
        """
        logger.info(f"Starting batch processing of {len(video_paths)} videos")
        
        batch_results = {
            'total_videos': len(video_paths),
            'successful_videos': 0,
            'failed_videos': 0,
            'video_results': {},
            'batch_summary': {}
        }
        
        for video_path in video_paths:
            try:
                video_name = Path(video_path).stem
                logger.info(f"Processing video {batch_results['successful_videos'] + batch_results['failed_videos'] + 1}/{len(video_paths)}: {video_name}")
                
                result = self.process_video(video_path, video_name)
                batch_results['video_results'][video_name] = result
                
                if result['overall_success']:
                    batch_results['successful_videos'] += 1
                else:
                    batch_results['failed_videos'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                batch_results['failed_videos'] += 1
                batch_results['video_results'][Path(video_path).stem] = {
                    'overall_success': False,
                    'error_message': str(e)
                }
        
        # Create batch summary
        batch_results['batch_summary'] = self._create_batch_summary(batch_results)
        
        # Save batch results
        self._save_batch_results(batch_results)
        
        logger.info(f"Batch processing completed: {batch_results['successful_videos']} successful, "
                   f"{batch_results['failed_videos']} failed")
        
        return batch_results
    
    def _create_final_summary(self, preprocessing_results: Dict, 
                            depth_results: Dict, output_results: Dict) -> Dict[str, Any]:
        """Create final pipeline summary."""
        summary = {
            'total_frames_extracted': preprocessing_results['quality_stats']['total_extracted'],
            'frames_accepted': preprocessing_results['quality_stats']['total_accepted'],
            'frames_processed_for_depth': len(depth_results['processed_frames']),
            'depth_quality_stats': depth_results.get('depth_stats', {}),
            'output_quality_stats': output_results.get('quality_stats', {}),
            'processing_time': {
                'preprocessing': 'N/A',  # Would need to track timing
                'depth_estimation': 'N/A',
                'output_management': 'N/A'
            },
            'file_counts': {
                'depth_maps': len(output_results.get('files', {}).get('depth_maps', [])),
                'previews': len(output_results.get('files', {}).get('previews', [])),
                'metadata_files': len(output_results.get('files', {}).get('metadata_files', []))
            }
        }
        
        return summary
    
    def _create_batch_summary(self, batch_results: Dict) -> Dict[str, Any]:
        """Create batch processing summary."""
        successful_results = [
            result for result in batch_results['video_results'].values()
            if result.get('overall_success', False)
        ]
        
        if not successful_results:
            return {
                'success_rate': 0.0,
                'total_frames_processed': 0,
                'average_quality': 0.0
            }
        
        total_frames = sum(
            result.get('frames_processed_for_depth', 0) 
            for result in successful_results
        )
        
        # Calculate average quality if available
        quality_scores = []
        for result in successful_results:
            output_stats = result.get('output_quality_stats', {})
            if 'mean_quality' in output_stats:
                quality_scores.append(output_stats['mean_quality'])
        
        summary = {
            'success_rate': batch_results['successful_videos'] / batch_results['total_videos'],
            'total_frames_processed': total_frames,
            'average_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            'quality_distribution': {
                'min_quality': min(quality_scores) if quality_scores else 0.0,
                'max_quality': max(quality_scores) if quality_scores else 0.0,
                'std_quality': 0.0  # Would need numpy for proper calculation
            }
        }
        
        return summary
    
    def _save_pipeline_results(self, video_name: str, results: Dict):
        """Save complete pipeline results."""
        results_file = self.output_dir / f"{video_name}_pipeline_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved: {results_file}")
    
    def _save_batch_results(self, batch_results: Dict):
        """Save batch processing results."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"batch_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        logger.info(f"Batch results saved: {results_file}")


def setup_environment_check():
    """Verify that the environment is properly set up."""
    logger.info("Checking environment setup...")
    
    try:
        setup = DepthEnvironmentSetup()
        if setup.verify_installation():
            logger.info("Environment setup verified successfully")
            return True
        else:
            logger.error("Environment setup verification failed")
            return False
    except Exception as e:
        logger.error(f"Environment check failed: {e}")
        return False


def main():
    """Main entry point for the depth estimation pipeline."""
    parser = argparse.ArgumentParser(
        description="Video Depth Estimation Pipeline - Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single video with default settings
    python main_pipeline.py video.mp4
    
    # Process video with custom output directory
    python main_pipeline.py video.mp4 --output-dir ./depth_results
    
    # Process multiple videos
    python main_pipeline.py video1.mp4 video2.mp4 video3.mp4
    
    # Process with custom frame interval and quality settings
    python main_pipeline.py video.mp4 --frame-interval 5 --enable-stabilization
    
    # Setup environment only
    python main_pipeline.py --setup-only
        """
    )
    
    # Input arguments
    parser.add_argument("videos", nargs="*", help="Input video file(s)")
    parser.add_argument("--output-dir", "-o", default="output", 
                       help="Output directory for all results")
    
    # Processing options
    parser.add_argument("--frame-interval", type=int, default=1,
                       help="Extract every nth frame")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum number of frames to process")
    parser.add_argument("--target-fps", type=float,
                       help="Target FPS for frame extraction")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                       help="Resize frames to specified dimensions")
    
    # Quality and processing options
    parser.add_argument("--enable-stabilization", action="store_true",
                       help="Enable video stabilization")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                       help="Blur detection threshold")
    parser.add_argument("--brightness-range", nargs=2, type=float, default=[20.0, 235.0],
                       help="Acceptable brightness range")
    
    # Depth estimation options
    parser.add_argument("--depth-model", default="depth-anything-v2-base",
                       choices=["depth-anything-v2-base", "depth-anything-v2-large"],
                       help="Depth estimation model")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for depth estimation")
    parser.add_argument("--max-resolution", nargs=2, type=int, default=[1024, 1024],
                       help="Maximum resolution for depth estimation")
    parser.add_argument("--disable-temporal-consistency", action="store_true",
                       help="Disable temporal consistency filtering")
    
    # Output options
    parser.add_argument("--depth-format", choices=["npy", "exr", "png"], default="npy",
                       help="Output format for depth maps")
    parser.add_argument("--no-archive", action="store_true",
                       help="Disable archiving of results")
    parser.add_argument("--keep-intermediates", action="store_true",
                       help="Keep intermediate processing files")
    
    # Environment setup
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup environment, don't process videos")
    parser.add_argument("--verify-setup", action="store_true",
                       help="Verify environment setup before processing")
    
    args = parser.parse_args()
    
    # Setup environment if requested
    if args.setup_only:
        logger.info("Setting up environment...")
        setup = DepthEnvironmentSetup()
        if setup.setup_environment():
            logger.info("Environment setup completed successfully")
            return 0
        else:
            logger.error("Environment setup failed")
            return 1
    
    # Verify setup if requested
    if args.verify_setup:
        if not setup_environment_check():
            logger.error("Environment verification failed. Run with --setup-only first.")
            return 1
    
    # Check for video inputs
    if not args.videos:
        parser.error("No video files provided. Use --setup-only for environment setup only.")
    
    # Validate video files
    video_paths = []
    for video_path in args.videos:
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            continue
        video_paths.append(video_path)
    
    if not video_paths:
        logger.error("No valid video files found")
        return 1
    
    # Create configurations
    processing_config = create_default_config()
    processing_config.frame_interval = args.frame_interval
    if args.max_frames:
        processing_config.max_frames = args.max_frames
    if args.target_fps:
        processing_config.target_fps = args.target_fps
    if args.resize:
        processing_config.resize_target = tuple(args.resize)
    processing_config.enable_stabilization = args.enable_stabilization
    processing_config.blur_threshold = args.blur_threshold
    processing_config.brightness_range = tuple(args.brightness_range)
    
    depth_config = create_default_depth_config()
    depth_config.model_name = args.depth_model
    depth_config.batch_size = args.batch_size
    depth_config.max_resolution = tuple(args.max_resolution)
    depth_config.apply_temporal_consistency = not args.disable_temporal_consistency
    depth_config.output_format = args.depth_format
    
    output_config = create_default_output_config()
    output_config.base_output_dir = args.output_dir
    output_config.depth_format = args.depth_format
    output_config.enable_archiving = not args.no_archive
    output_config.keep_intermediate_files = args.keep_intermediates
    
    # Initialize pipeline
    pipeline = DepthEstimationMaster(
        processing_config=processing_config,
        depth_config=depth_config,
        output_config=output_config,
        output_dir=args.output_dir
    )
    
    # Process videos
    if len(video_paths) == 1:
        # Single video
        results = pipeline.process_video(video_paths[0])
        
        if results['overall_success']:
            logger.info("Processing completed successfully!")
            print(f"\nResults Summary:")
            print(f"Video: {results['video_name']}")
            print(f"Frames processed: {results['frames_processed_for_depth']}")
            print(f"Output directory: {args.output_dir}")
            return 0
        else:
            logger.error(f"Processing failed: {results['error_message']}")
            return 1
    else:
        # Multiple videos
        results = pipeline.process_multiple_videos(video_paths)
        
        print(f"\nBatch Processing Summary:")
        print(f"Total videos: {results['total_videos']}")
        print(f"Successful: {results['successful_videos']}")
        print(f"Failed: {results['failed_videos']}")
        print(f"Success rate: {results['batch_summary']['success_rate']:.1%}")
        print(f"Total frames processed: {results['batch_summary']['total_frames_processed']}")
        print(f"Output directory: {args.output_dir}")
        
        return 0 if results['successful_videos'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
