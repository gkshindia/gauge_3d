"""
Video frame extraction module for 3D reconstruction pipeline.
Handles video input, frame extraction, and preprocessing.
"""

import cv2
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
from tqdm import tqdm


class VideoFrameExtractor:
    """Extract frames from video files for 3D reconstruction processing."""
    
    def __init__(self, output_dir: str = "output/frames"):
        """
        Initialize the frame extractor.
        
        Args:
            output_dir: Directory to save extracted frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_video_info(self, video_path: str) -> dict:
        """
        Get comprehensive information about the video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        info = {
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
    
    def extract_frames(self, 
                      video_path: str, 
                      frame_interval: int = 1,
                      start_frame: int = 0,
                      end_frame: Optional[int] = None,
                      resize: Optional[Tuple[int, int]] = None,
                      quality: int = 95) -> List[str]:
        """
        Extract frames from video with various options.
        
        Args:
            video_path: Path to the input video file
            frame_interval: Extract every nth frame (1 = all frames)
            start_frame: Frame number to start extraction
            end_frame: Frame number to end extraction (None = end of video)
            resize: Tuple of (width, height) to resize frames
            quality: JPEG quality for saved frames (0-100)
            
        Returns:
            List of paths to extracted frame files
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video info
        video_info = self.get_video_info(video_path)
        print("Video Info:")
        print(f"  Resolution: {video_info['width']}x{video_info['height']}")
        print(f"  FPS: {video_info['fps']:.2f}")
        print(f"  Total frames: {video_info['total_frames']}")
        print(f"  Duration: {video_info['duration']:.2f} seconds")
        
        # Set frame range
        if end_frame is None:
            end_frame = video_info['total_frames']
        
        # Create output subdirectory for this video
        video_name = Path(video_path).stem
        frame_output_dir = self.output_dir / video_name
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        extracted_files = []
        frame_count = 0
        saved_count = 0
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        total_to_extract = (end_frame - start_frame) // frame_interval
        
        with tqdm(total=total_to_extract, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret or frame_count + start_frame >= end_frame:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    # Resize frame if specified
                    if resize:
                        frame = cv2.resize(frame, resize)
                    
                    # Save frame
                    frame_filename = f"frame_{frame_count + start_frame:06d}.jpg"
                    frame_path = frame_output_dir / frame_filename
                    
                    # Set JPEG quality
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    extracted_files.append(str(frame_path))
                    saved_count += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        print("Extraction complete!")
        print(f"  Extracted {saved_count} frames")
        print(f"  Saved to: {frame_output_dir}")
        print(f"  Frame interval: {frame_interval}")
        
        return extracted_files
    
    def extract_key_frames(self, 
                          video_path: str, 
                          num_frames: int = 50,
                          method: str = "uniform") -> List[str]:
        """
        Extract key frames using different sampling methods.
        
        Args:
            video_path: Path to the input video file
            num_frames: Number of frames to extract
            method: Sampling method ("uniform", "adaptive")
            
        Returns:
            List of paths to extracted frame files
        """
        video_info = self.get_video_info(video_path)
        total_frames = video_info['total_frames']
        
        if method == "uniform":
            # Extract frames uniformly distributed across the video
            frame_interval = max(1, total_frames // num_frames)
            return self.extract_frames(
                video_path, 
                frame_interval=frame_interval,
                end_frame=min(total_frames, num_frames * frame_interval)
            )
        
        elif method == "adaptive":
            # Simple adaptive method - could be enhanced with motion detection
            # For now, use uniform sampling with slight randomization
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            cap = cv2.VideoCapture(video_path)
            video_name = Path(video_path).stem
            frame_output_dir = self.output_dir / video_name
            frame_output_dir.mkdir(parents=True, exist_ok=True)
            
            extracted_files = []
            
            with tqdm(total=num_frames, desc="Extracting key frames") as pbar:
                for i, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame_filename = f"keyframe_{i:04d}_frame_{frame_idx:06d}.jpg"
                        frame_path = frame_output_dir / frame_filename
                        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        extracted_files.append(str(frame_path))
                    
                    pbar.update(1)
            
            cap.release()
            return extracted_files
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def clean_output_dir(self, video_name: Optional[str] = None):
        """
        Clean the output directory.
        
        Args:
            video_name: If specified, only clean files for this video
        """
        if video_name:
            target_dir = self.output_dir / video_name
            if target_dir.exists():
                import shutil
                shutil.rmtree(target_dir)
                print(f"Cleaned frames for video: {video_name}")
        else:
            if self.output_dir.exists():
                import shutil
                shutil.rmtree(self.output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                print("Cleaned all extracted frames")


def main():
    """Command-line interface for frame extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from video for 3D reconstruction")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output-dir", "-o", default="output/frames", 
                       help="Output directory for extracted frames")
    parser.add_argument("--frame-interval", "-i", type=int, default=1,
                       help="Extract every nth frame (default: 1)")
    parser.add_argument("--start-frame", "-s", type=int, default=0,
                       help="Frame number to start extraction")
    parser.add_argument("--end-frame", "-e", type=int, default=None,
                       help="Frame number to end extraction")
    parser.add_argument("--resize", "-r", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                       help="Resize frames to specified dimensions")
    parser.add_argument("--quality", "-q", type=int, default=95,
                       help="JPEG quality (0-100, default: 95)")
    parser.add_argument("--key-frames", "-k", type=int, metavar="NUM",
                       help="Extract NUM key frames instead of regular extraction")
    parser.add_argument("--method", "-m", choices=["uniform", "adaptive"], default="uniform",
                       help="Key frame sampling method")
    parser.add_argument("--clean", action="store_true",
                       help="Clean output directory before extraction")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = VideoFrameExtractor(args.output_dir)
    
    # Clean if requested
    if args.clean:
        extractor.clean_output_dir()
    
    try:
        # Extract frames
        if args.key_frames:
            extracted_files = extractor.extract_key_frames(
                args.video_path, 
                num_frames=args.key_frames,
                method=args.method
            )
        else:
            resize_tuple = tuple(args.resize) if args.resize else None
            extracted_files = extractor.extract_frames(
                args.video_path,
                frame_interval=args.frame_interval,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                resize=resize_tuple,
                quality=args.quality
            )
        
        print(f"\nSuccessfully extracted {len(extracted_files)} frames")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
