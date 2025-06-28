#!/usr/bin/env python3
"""
Gauge 3D Project Entry Point

Quick commands for common operations:
- Extract frames from video
- Analyze workspace
- Get video information

Run with --help for more options.
"""

import sys
import subprocess
from pathlib import Path


def run_main(*args):
    """Run the main application with arguments."""
    cmd = [sys.executable, "main.py"] + list(args)
    return subprocess.run(cmd, cwd=Path(__file__).parent)


def run_utils(*args):
    """Run utilities with arguments."""
    cmd = [sys.executable, "utils.py"] + list(args)
    return subprocess.run(cmd, cwd=Path(__file__).parent)


def main():
    """Simple CLI dispatcher."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gauge 3D Project - Quick Commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Examples:
    # Extract frames from video
    python run.py extract video.mp4
    
    # Extract key frames
    python run.py extract video.mp4 --key-frames 30
    
    # Analyze workspace
    python run.py analyze
    
    # List all videos and extractions
    python run.py list
    
    # Get video info
    python run.py info video.mp4
    
    # Run full main.py with custom options
    python run.py main video.mp4 --frame-interval 5 --resize 1920 1080
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command - simplified interface
    extract_parser = subparsers.add_parser('extract', help='Extract frames from video')
    extract_parser.add_argument('video_path', help='Path to video file')
    extract_parser.add_argument('--key-frames', '-k', type=int, help='Extract N key frames')
    extract_parser.add_argument('--interval', '-i', type=int, default=30, help='Frame interval')
    extract_parser.add_argument('--resize', '-r', nargs=2, type=int, help='Resize WIDTH HEIGHT')
    extract_parser.add_argument('--clean', action='store_true', help='Clean output first')
    
    # Analyze command
    subparsers.add_parser('analyze', help='Analyze workspace')
    
    # List command
    subparsers.add_parser('list', help='List videos and extractions')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get video information')
    info_parser.add_argument('video_path', help='Path to video file')
    
    # Main command - pass through to main.py
    main_parser = subparsers.add_parser('main', help='Run main.py with full options')
    main_parser.add_argument('args', nargs='*', help='Arguments to pass to main.py')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        main_args = [args.video_path]
        
        if args.key_frames:
            main_args.extend(['--key-frames', str(args.key_frames)])
        else:
            main_args.extend(['--frame-interval', str(args.interval)])
        
        if args.resize:
            main_args.extend(['--resize'] + [str(x) for x in args.resize])
        
        if args.clean:
            main_args.append('--clean')
        
        main_args.append('--verbose')
        
        return run_main(*main_args)
    
    elif args.command == 'analyze':
        return run_utils('analyze')
    
    elif args.command == 'list':
        return run_utils('list')
    
    elif args.command == 'info':
        return run_utils('info', args.video_path)
    
    elif args.command == 'main':
        return run_main(*args.args)
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    result = main()
    if result:
        sys.exit(result.returncode)
