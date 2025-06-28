#!/usr/bin/env python3
"""
Phase 1 Integration - Complete Video to Depth Pipeline
Complete integration of video processing and depth estimation using DepthAnyVideo
"""

import os
import sys
import argparse
from pathlib import Path
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.video_processor import VideoFrameExtractor as VideoProcessor
from depth_estimation.depth_pipeline import DepthEstimationPipeline, create_default_depth_config
from depth_estimation.setup_environment import DepthAnyVideoSetup

def main():
    """Complete Phase 1 pipeline: Video -> Frames -> Depth Maps"""
    
    parser = argparse.ArgumentParser(description="Complete video to depth estimation pipeline")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--fps", type=int, help="Target FPS for frame extraction")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to extract")
    parser.add_argument("--depth-model", default="depth-anything-v2-base", 
                       choices=["depth-anything-v2-small", "depth-anything-v2-base", "depth-anything-v2-large"],
                       help="Depth estimation model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for depth estimation")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--depth-format", choices=["npy", "png"], default="npy", help="Depth map output format")
    
    args = parser.parse_args()
    
    # Validate input
    input_video = Path(args.input_video)
    if not input_video.exists():
        print(f"❌ Input video not found: {input_video}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    video_name = input_video.stem
    
    print("🎬 Phase 1: Complete Video to Depth Pipeline")
    print(f"Input: {input_video}")
    print(f"Output: {output_dir}")
    print(f"Video: {video_name}")
    print("-" * 50)
    
    # Step 1: Environment Setup
    if not args.skip_setup:
        print("\n📋 Step 1: Environment Setup")
        setup = DepthAnyVideoSetup()
        
        if not setup.verify_installation():
            print("Setting up environment...")
            setup.setup_complete_environment()
            
            if not setup.verify_installation():
                print("❌ Environment setup failed!")
                sys.exit(1)
        
        print("✅ Environment ready")
    
    # Step 2: Video Processing
    print("\n🎞️  Step 2: Video Frame Extraction")
    video_processor = VideoProcessor(str(output_dir / "frames"))
    
    # Configure frame extraction parameters
    extract_params = {
        'frame_interval': 1,
        'quality': 95
    }
    
    # Calculate frame interval based on target FPS if provided
    if args.fps:
        # We'll extract every nth frame to achieve target FPS
        # This is a simplified approach - more sophisticated would analyze video FPS
        video_fps = 30  # Assume 30fps source, could detect this
        extract_params['frame_interval'] = max(1, video_fps // args.fps)
    
    try:
        # Extract frames
        frame_paths = video_processor.extract_frames(
            str(input_video), 
            **extract_params
        )
        
        # Limit frames if requested
        if args.max_frames and len(frame_paths) > args.max_frames:
            frame_paths = frame_paths[:args.max_frames]
        
        print(f"✅ Extracted {len(frame_paths)} frames")
        
        # Create frame info for compatibility
        frame_info = {
            'frame_count': len(frame_paths),
            'resolution': 'Unknown',  # VideoFrameExtractor doesn't return this
            'fps': args.fps if args.fps else 'Unknown'
        }
        
    except Exception as e:
        print(f"❌ Frame extraction failed: {e}")
        sys.exit(1)
    
    # Step 3: Depth Estimation
    print("\n🔍 Step 3: Depth Estimation")
    
    # Create depth configuration
    depth_config = create_default_depth_config()
    depth_config.model_name = args.depth_model
    depth_config.batch_size = args.batch_size
    depth_config.output_format = args.depth_format
    
    # Get frame paths
    frames_dir = output_dir / "frames"
    all_frame_paths = sorted(frames_dir.glob("**/*.jpg"))
    
    if not all_frame_paths:
        print(f"❌ No frames found in {frames_dir}")
        sys.exit(1)
    
    # Use the extracted frame paths
    frame_paths = all_frame_paths
    
    print(f"Processing {len(frame_paths)} frames with {args.depth_model}")
    
    try:
        # Initialize depth estimation pipeline
        depth_pipeline = DepthEstimationPipeline(
            depth_config, 
            str(output_dir)
        )
        
        # Process frames
        start_time = time.time()
        results = depth_pipeline.process_video_sequence(
            [str(p) for p in frame_paths], 
            video_name
        )
        processing_time = time.time() - start_time
        
        print(f"✅ Depth estimation completed in {processing_time:.1f}s")
        print(f"   Processed: {len(results['processed_frames'])} frames")
        print(f"   Depth range: {results['depth_stats']['min_depth']:.3f} - {results['depth_stats']['max_depth']:.3f}")
        print(f"   Average depth: {results['depth_stats']['avg_depth']:.3f}")
        
    except Exception as e:
        print(f"❌ Depth estimation failed: {e}")
        sys.exit(1)
    
    # Step 4: Summary
    print("\n📊 Phase 1 Complete!")
    print("Generated outputs:")
    print(f"  📁 Frames: {output_dir}/frames/{video_name}/")
    print(f"  📁 Depth maps: {output_dir}/depth_maps/{video_name}/")
    print(f"  📁 Previews: {output_dir}/depth_previews/{video_name}/")
    print(f"  📄 Stats: {output_dir}/depth_stats/{video_name}_depth_results.json")
    
    # Performance stats
    total_frames = len(frame_paths)
    fps_processed = total_frames / processing_time if processing_time > 0 else 0
    print(f"\n⚡ Performance: {fps_processed:.2f} FPS processing speed")
    
    print("\n🎯 Phase 1 completed successfully!")
    print("Ready for Phase 2: 4D Gaussian Splatting")


if __name__ == "__main__":
    main()
