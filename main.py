#!/usr/bin/env python3
"""
Gauge 3D - Video to 3D Reconstruction and Distance Measurement

This application processes video files to extract frames, perform 3D reconstruction,
generate point clouds, and measure distances between objects.

Usage:
    python main.py <video_path> [options]
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_processor import HybridVideoProcessor


def main():
    """Main entry point for the Gauge 3D application."""
    parser = argparse.ArgumentParser(
        description="Gauge 3D - Convert video to 3D reconstruction and measure distances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract all frames from a video
    python main.py video.mp4
    
    # Extract every 10th frame
    python main.py video.mp4 --frame-interval 10
    
    # Extract 50 key frames uniformly distributed
    python main.py video.mp4 --key-frames 50
    
    # Extract frames with custom output directory and resize
    python main.py video.mp4 --output-dir ./frames --resize 1920 1080
    
    # Extract frames from specific time range
    python main.py video.mp4 --start-frame 100 --end-frame 500
        """
    )
    
    # Required arguments
    parser.add_argument("video_path", help="Path to the input video file")
    
    # Frame extraction options
    parser.add_argument("--output-dir", "-o", default="output/frames", 
                       help="Output directory for extracted frames (default: output/frames)")
    parser.add_argument("--frame-interval", "-i", type=int, default=30,
                       help="Extract every nth frame (default: 30)")
    parser.add_argument("--start-frame", "-s", type=int, default=0,
                       help="Frame number to start extraction (default: 0)")
    parser.add_argument("--end-frame", "-e", type=int, default=None,
                       help="Frame number to end extraction (default: end of video)")
    parser.add_argument("--resize", "-r", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                       help="Resize frames to specified dimensions")
    parser.add_argument("--quality", "-q", type=int, default=95,
                       help="JPEG quality for saved frames (0-100, default: 95)")
    
    # Key frame extraction
    parser.add_argument("--key-frames", "-k", type=int, metavar="NUM",
                       help="Extract NUM key frames instead of regular extraction")
    parser.add_argument("--method", "-m", choices=["uniform", "adaptive"], default="uniform",
                       help="Key frame sampling method (default: uniform)")
    
    # Processing method options
    parser.add_argument("--engine", "-x", choices=["auto", "opencv", "ffmpeg"], default="auto",
                       help="Processing engine to use (default: auto)")
    parser.add_argument("--prefer-opencv", action="store_true",
                       help="Prefer OpenCV over FFmpeg when both available")
    
    # Utility options
    parser.add_argument("--clean", action="store_true",
                       help="Clean output directory before extraction")
    parser.add_argument("--info", action="store_true",
                       help="Show video information only, don't extract frames")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--benchmark", action="store_true",
                       help="Benchmark available extraction methods")
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    
    if not video_path.is_file():
        print(f"Error: Path is not a file: {video_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check file extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    if video_path.suffix.lower() not in valid_extensions:
        print(f"Warning: File extension '{video_path.suffix}' may not be supported")
        print(f"Supported extensions: {', '.join(valid_extensions)}")
    
    try:
        # Initialize processor
        prefer_ffmpeg = not args.prefer_opencv
        processor = HybridVideoProcessor(args.output_dir, prefer_ffmpeg=prefer_ffmpeg)
        
        # Show video info if requested
        if args.info:
            video_info = processor.get_video_info(str(video_path))
            print("Video Information:")
            print("=" * 50)
            print(f"File: {video_path}")
            print(f"Resolution: {video_info['width']}x{video_info['height']}")
            print(f"FPS: {video_info['fps']:.2f}")
            print(f"Total frames: {video_info['total_frames']}")
            print(f"Duration: {video_info['duration']:.2f} seconds")
            print(f"Codec: {video_info['fourcc']}")
            return
        
        # Benchmark if requested
        if args.benchmark:
            print("Benchmarking extraction methods...")
            results = processor.benchmark_methods(str(video_path), args.frame_interval)
            
            print("\nBenchmark Results:")
            print("=" * 50)
            for method, result in results.items():
                print(f"{method.upper()}:")
                if 'error' in result:
                    print(f"  Error: {result['error']}")
                else:
                    print(f"  Time: {result['time']:.2f}s")
                    print(f"  Frames: {result['frames']}")
                    print(f"  Processing rate: {result['fps']:.1f} frames/sec")
                print()
            return
        
        # Clean output directory if requested
        if args.clean:
            print("Cleaning output directory...")
            processor.opencv_processor.clean_output_dir()
        
        print(f"Processing video: {video_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"Using engine: {args.engine}")
        
        # Extract frames based on mode
        if args.key_frames:
            print(f"Extracting {args.key_frames} key frames using {args.method} method...")
            extracted_files = processor.extract_key_frames(
                str(video_path), 
                num_frames=args.key_frames,
                method=args.method if args.engine == "auto" else args.engine
            )
        else:
            print(f"Extracting frames with interval {args.frame_interval}...")
            resize_tuple = tuple(args.resize) if args.resize else None
            extracted_files = processor.extract_frames(
                str(video_path),
                frame_interval=args.frame_interval,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                resize=resize_tuple,
                quality=args.quality,
                method=args.engine
            )
        
        print("\n" + "=" * 50)
        print("Frame extraction completed successfully!")
        print(f"Total frames extracted: {len(extracted_files)}")
        
        if args.verbose and extracted_files:
            print("\nFirst 5 extracted files:")
            for file_path in extracted_files[:5]:
                print(f"  {file_path}")
            if len(extracted_files) > 5:
                print(f"  ... and {len(extracted_files) - 5} more files")
        
        print("\nNext steps:")
        print("- Use the extracted frames for 3D reconstruction")
        print("- Generate point clouds from the reconstructed data")
        print("- Measure distances between objects in 3D space")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
