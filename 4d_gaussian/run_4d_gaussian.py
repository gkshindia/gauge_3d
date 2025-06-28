#!/usr/bin/env python3
"""
Step 2 Integration - 4D Gaussian Splatting Pipeline
Complete integration of depth maps to 4D Gaussian reconstruction
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import 4D Gaussian modules
from config.gaussian_config import get_default_config, get_data_config
from setup.environment_setup import GaussianEnvironmentSetup
from data_preparation.data_converter import DepthToGaussianConverter
from gaussian_generation.gaussian_initializer import GaussianInitializer

def setup_logging(output_dir: str):
    """Setup logging for the pipeline"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"phase2_{int(time.time())}.log")
        ]
    )

def run_environment_setup() -> bool:
    """Run environment setup for 4D Gaussian Splatting"""
    logger = logging.getLogger(__name__)
    logger.info("Starting 4D Gaussian environment setup...")
    
    setup = GaussianEnvironmentSetup()
    success = setup.setup_environment()
    
    if success:
        logger.info("✅ 4D Gaussian environment setup completed")
    else:
        logger.error("❌ 4D Gaussian environment setup failed")
    
    return success

def run_data_preparation(video_name: str, config: dict) -> bool:
    """Run data preparation phase"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting data preparation for {video_name}...")
    
    try:
        # Initialize data converter
        converter = DepthToGaussianConverter(
            frames_dir=config["frames_dir"],
            depth_maps_dir=config["depth_maps_dir"],
            output_dir=config["gaussian_output_dir"],
            config=config
        )
        
        # Process video sequence
        result = converter.process_video_sequence(video_name)
        
        logger.info(f"✅ Data preparation completed: {result['num_frames']} frames processed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        return False

def run_gaussian_initialization(video_name: str, config: dict) -> bool:
    """Run Gaussian initialization phase"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Gaussian initialization for {video_name}...")
    
    try:
        # Initialize Gaussian initializer
        initializer = GaussianInitializer(config)
        
        # Load point clouds
        point_clouds_dir = Path(config["gaussian_output_dir"]) / "point_clouds" / video_name
        
        if not point_clouds_dir.exists():
            raise ValueError(f"Point clouds directory not found: {point_clouds_dir}")
        
        # Load point cloud files
        pc_files = sorted(list(point_clouds_dir.glob("*.npy")))
        point_clouds = []
        
        for pc_file in pc_files:
            pc = np.load(pc_file)
            point_clouds.append(pc)
        
        logger.info(f"Loaded {len(point_clouds)} point clouds")
        
        # Initialize Gaussians
        gaussians_sequence = initializer.initialize_temporal_gaussians(
            point_clouds, config.get("temporal_window", 10)
        )
        
        # Create correspondences
        correspondences = initializer.create_temporal_correspondences(gaussians_sequence)
        
        # Save results
        output_dir = Path(config["gaussian_output_dir"]) / "gaussian_init" / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Gaussians for each frame
        for i, gaussians in enumerate(gaussians_sequence):
            output_file = output_dir / f"frame_{i:06d}_gaussians.npz"
            initializer.save_gaussians(gaussians, str(output_file))
        
        # Save correspondences
        correspondences_file = output_dir / "temporal_correspondences.json"
        with open(correspondences_file, 'w') as f:
            json.dump(correspondences, f, indent=2)
        
        logger.info(f"✅ Gaussian initialization completed: {len(gaussians_sequence)} frames")
        return True
        
    except Exception as e:
        logger.error(f"❌ Gaussian initialization failed: {e}")
        return False

def validate_phase1_outputs(video_name: str, frames_dir: str, depth_maps_dir: str) -> bool:
    """Validate that Phase 1 outputs exist"""
    logger = logging.getLogger(__name__)
    
    frames_path = Path(frames_dir) / video_name
    depth_path = Path(depth_maps_dir) / video_name
    
    if not frames_path.exists():
        logger.error(f"Frames directory not found: {frames_path}")
        return False
    
    if not depth_path.exists():
        logger.error(f"Depth maps directory not found: {depth_path}")
        return False
    
    frame_files = list(frames_path.glob("*.jpg"))
    depth_files = list(depth_path.glob("*_depth.npy"))
    
    if not frame_files:
        logger.error(f"No frame files found in {frames_path}")
        return False
    
    if not depth_files:
        logger.error(f"No depth files found in {depth_path}")
        return False
    
    logger.info(f"Found {len(frame_files)} frames and {len(depth_files)} depth maps")
    return True

def main():
    """Main entry point for Phase 2 pipeline"""
    
    parser = argparse.ArgumentParser(description="Phase 2: 4D Gaussian Splatting Pipeline")
    parser.add_argument("video_name", help="Name of video to process")
    parser.add_argument("--frames-dir", default="output/frames", help="Frames directory")
    parser.add_argument("--depth-maps-dir", default="output/depth_maps", help="Depth maps directory")
    parser.add_argument("--output-dir", default="output/gaussian_reconstruction", help="Output directory")
    parser.add_argument("--setup-only", action="store_true", help="Only run environment setup")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--max-gaussians", type=int, default=1000000, help="Maximum number of Gaussians")
    parser.add_argument("--temporal-window", type=int, default=10, help="Temporal window size")
    parser.add_argument("--use-colmap", action="store_true", help="Use COLMAP for pose estimation")
    parser.add_argument("--quality", choices=["preview", "standard", "high", "ultra"], 
                       default="standard", help="Quality preset for point cloud density")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("PHASE 2: 4D GAUSSIAN SPLATTING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Video: {args.video_name}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Step 1: Environment Setup
        if not args.skip_setup:
            logger.info("Step 1: Environment Setup")
            if not run_environment_setup():
                logger.error("Environment setup failed")
                sys.exit(1)
            
            if args.setup_only:
                logger.info("Setup completed. Exiting as requested.")
                sys.exit(0)
        else:
            logger.info("Skipping environment setup")
        
        # Step 2: Validate Phase 1 outputs
        logger.info("Step 2: Validating Phase 1 outputs")
        if not validate_phase1_outputs(args.video_name, args.frames_dir, args.depth_maps_dir):
            logger.error("Phase 1 outputs validation failed")
            sys.exit(1)
        
        # Create configuration
        config = {
            "frames_dir": args.frames_dir,
            "depth_maps_dir": args.depth_maps_dir,
            "gaussian_output_dir": args.output_dir,
            "max_gaussians": args.max_gaussians,
            "temporal_window": args.temporal_window,
            "use_colmap": args.use_colmap,
            "quality": args.quality,
            "filter_point_clouds": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        # Step 3: Data Preparation
        logger.info("Step 3: Data Preparation (Depth to Point Clouds)")
        if not run_data_preparation(args.video_name, config):
            logger.error("Data preparation failed")
            sys.exit(1)
        
        # Step 4: Gaussian Initialization
        logger.info("Step 4: Gaussian Initialization")
        if not run_gaussian_initialization(args.video_name, config):
            logger.error("Gaussian initialization failed")
            sys.exit(1)
        
        # Success!
        logger.info("=" * 80)
        logger.info("✅ PHASE 2 PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("Next steps:")
        logger.info("  - Run Phase 3: Temporal optimization")
        logger.info("  - Run Phase 4: Rendering and visualization")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Import numpy here to avoid import issues during setup
    import numpy as np
    import torch
    
    main()
