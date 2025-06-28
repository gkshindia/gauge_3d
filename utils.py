#!/usr/bin/env python3
"""
Utility functions for the Gauge 3D project.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_processor import VideoFrameExtractor


def analyze_workspace(base_dir: str = ".") -> Dict[str, Any]:
    """
    Analyze the current workspace and provide a summary.
    
    Args:
        base_dir: Base directory to analyze
        
    Returns:
        Dictionary containing workspace analysis
    """
    base_path = Path(base_dir)
    
    # Find video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(base_path.glob(f"**/*{ext}")))
        video_files.extend(list(base_path.glob(f"**/*{ext.upper()}")))
    
    # Find extracted frames
    frames_dir = base_path / "output" / "frames"
    extracted_sessions = []
    
    if frames_dir.exists():
        for session_dir in frames_dir.iterdir():
            if session_dir.is_dir():
                frame_count = len(list(session_dir.glob("*.jpg"))) + len(list(session_dir.glob("*.png")))
                extracted_sessions.append({
                    'name': session_dir.name,
                    'frame_count': frame_count,
                    'path': str(session_dir)
                })
    
    return {
        'video_files': [str(f) for f in video_files],
        'video_count': len(video_files),
        'extracted_sessions': extracted_sessions,
        'total_extracted_frames': sum(s['frame_count'] for s in extracted_sessions)
    }


def get_video_summary(video_path: str) -> Dict[str, Any]:
    """
    Get a comprehensive summary of a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video summary
    """
    extractor = VideoFrameExtractor()
    
    try:
        info = extractor.get_video_info(video_path)
        
        # Calculate additional metrics
        total_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
        
        # Estimate processing time (rough approximation)
        estimated_processing_time = info['total_frames'] / 10  # 10 fps processing estimate
        
        return {
            'file_path': video_path,
            'file_size_mb': round(total_size_mb, 2),
            'resolution': f"{info['width']}x{info['height']}",
            'width': info['width'],
            'height': info['height'],
            'fps': round(info['fps'], 2),
            'total_frames': info['total_frames'],
            'duration_seconds': round(info['duration'], 2),
            'duration_formatted': f"{int(info['duration'] // 60)}:{int(info['duration'] % 60):02d}",
            'estimated_processing_time_seconds': round(estimated_processing_time, 1),
            'codec': info['fourcc']
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    """CLI interface for utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gauge 3D Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze workspace')
    analyze_parser.add_argument('--dir', '-d', default='.', help='Directory to analyze')
    analyze_parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    # Video info command
    info_parser = subparsers.add_parser('info', help='Get video information')
    info_parser.add_argument('video_path', help='Path to video file')
    info_parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    # List command
    subparsers.add_parser('list', help='List available videos and extractions')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analysis = analyze_workspace(args.dir)
        
        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print("Workspace Analysis")
            print("=" * 50)
            print(f"Videos found: {analysis['video_count']}")
            for video in analysis['video_files']:
                print(f"  - {video}")
            
            print(f"\nExtracted sessions: {len(analysis['extracted_sessions'])}")
            for session in analysis['extracted_sessions']:
                print(f"  - {session['name']}: {session['frame_count']} frames")
            
            print(f"\nTotal extracted frames: {analysis['total_extracted_frames']}")
    
    elif args.command == 'info':
        summary = get_video_summary(args.video_path)
        
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            if 'error' in summary:
                print(f"Error: {summary['error']}")
                sys.exit(1)
            
            print("Video Information")
            print("=" * 50)
            print(f"File: {summary['file_path']}")
            print(f"Size: {summary['file_size_mb']} MB")
            print(f"Resolution: {summary['resolution']}")
            print(f"FPS: {summary['fps']}")
            print(f"Duration: {summary['duration_formatted']} ({summary['duration_seconds']}s)")
            print(f"Total frames: {summary['total_frames']}")
            print(f"Estimated processing time: {summary['estimated_processing_time_seconds']}s")
    
    elif args.command == 'list':
        analysis = analyze_workspace()
        
        print("Available Videos:")
        print("-" * 30)
        for video in analysis['video_files']:
            try:
                summary = get_video_summary(video)
                if 'error' not in summary:
                    print(f"{video}")
                    print(f"  {summary['resolution']} @ {summary['fps']} fps, {summary['duration_formatted']}")
                else:
                    print(f"{video} (error reading)")
            except Exception:
                print(f"{video} (error reading)")
        
        if analysis['extracted_sessions']:
            print("\nExtracted Frame Sessions:")
            print("-" * 30)
            for session in analysis['extracted_sessions']:
                print(f"{session['name']}: {session['frame_count']} frames")
                print(f"  Path: {session['path']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
