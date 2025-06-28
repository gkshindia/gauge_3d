"""
Video frame extraction module for 3D reconstruction pipeline.
Handles video input, frame extraction, and preprocessing.
"""

import cv2
import os
import sys
import subprocess
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


class FFmpegExtractor:
    """FFmpeg-based frame extraction for high-performance processing."""
    
    def __init__(self, output_dir: str = "output/frames"):
        """
        Initialize the FFmpeg extractor.
        
        Args:
            output_dir: Directory to save extracted frames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if ffmpeg is available
        if not self._check_ffmpeg():
            raise RuntimeError("FFmpeg not found. Please install FFmpeg or use OpenCV mode.")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available in the system."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video information using FFprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Calculate frame count and duration
            fps = eval(video_stream.get('r_frame_rate', '30/1'))  # Handle fraction
            duration = float(data['format']['duration'])
            total_frames = int(fps * duration)
            
            return {
                'total_frames': total_frames,
                'fps': fps,
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'duration': duration,
                'fourcc': video_stream.get('codec_name', 'unknown')
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to get video info: {e}")
    
    def extract_frames(self, 
                      video_path: str, 
                      frame_interval: int = 1,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      resize: Optional[Tuple[int, int]] = None,
                      quality: int = 95) -> List[str]:
        """
        Extract frames using FFmpeg.
        
        Args:
            video_path: Path to the input video file
            frame_interval: Extract every nth frame
            start_time: Start time in seconds
            end_time: End time in seconds
            resize: Tuple of (width, height) to resize frames
            quality: JPEG quality (0-100)
            
        Returns:
            List of paths to extracted frame files
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video info
        video_info = self.get_video_info(video_path)
        print("Video Info (FFmpeg):")
        print(f"  Resolution: {video_info['width']}x{video_info['height']}")
        print(f"  FPS: {video_info['fps']:.2f}")
        print(f"  Total frames: {video_info['total_frames']}")
        print(f"  Duration: {video_info['duration']:.2f} seconds")
        
        # Create output directory
        video_name = Path(video_path).stem
        frame_output_dir = self.output_dir / video_name
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-i', video_path]
        
        # Add time range if specified
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        if end_time is not None:
            cmd.extend(['-t', str(end_time - (start_time or 0))])
        
        # Frame selection
        if frame_interval > 1:
            cmd.extend(['-vf', f'select=not(mod(n\\,{frame_interval}))'])
        
        # Resize if specified
        if resize:
            scale_filter = f'scale={resize[0]}:{resize[1]}'
            if frame_interval > 1:
                cmd[-1] = f'select=not(mod(n\\,{frame_interval})),{scale_filter}'
            else:
                cmd.extend(['-vf', scale_filter])
        
        # Quality and output format
        cmd.extend(['-q:v', str(int((100 - quality) / 4))])  # Convert to FFmpeg scale
        cmd.extend(['-vsync', 'vfr'])  # Variable frame rate to handle frame selection
        
        # Output pattern
        output_pattern = str(frame_output_dir / 'frame_%06d.jpg')
        cmd.append(output_pattern)
        
        # Overwrite existing files
        cmd.insert(1, '-y')
        
        print("Running FFmpeg extraction...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Run FFmpeg with progress
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            # Monitor progress (basic)
            stderr_output = ""
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    stderr_output += output
                    # Simple progress indication
                    if "frame=" in output:
                        print(".", end="", flush=True)
            
            print()  # New line after progress dots
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr_output)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg extraction failed: {e.stderr}")
        
        # Get list of extracted files
        extracted_files = sorted(frame_output_dir.glob("frame_*.jpg"))
        extracted_paths = [str(f) for f in extracted_files]
        
        print("Extraction complete!")
        print(f"  Extracted {len(extracted_paths)} frames")
        print(f"  Saved to: {frame_output_dir}")
        print(f"  Frame interval: {frame_interval}")
        
        return extracted_paths
    
    def extract_key_frames(self, 
                          video_path: str, 
                          num_frames: int = 50) -> List[str]:
        """
        Extract key frames using FFmpeg's scene detection.
        
        Args:
            video_path: Path to the input video file
            num_frames: Maximum number of frames to extract
            
        Returns:
            List of paths to extracted frame files
        """
        video_info = self.get_video_info(video_path)
        video_name = Path(video_path).stem
        frame_output_dir = self.output_dir / video_name
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use scene detection to find key frames
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', 'select=gt(scene\\,0.3),scale=iw:ih',
            '-vsync', 'vfr',
            '-q:v', '2',
            str(frame_output_dir / 'keyframe_%04d.jpg')
        ]
        
        print("Extracting key frames using FFmpeg scene detection...")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Fallback to uniform sampling if scene detection fails
            print("Scene detection failed, falling back to uniform sampling...")
            interval = max(1, video_info['total_frames'] // num_frames)
            return self.extract_frames(video_path, frame_interval=interval)
        
        # Get extracted files and limit to requested number
        extracted_files = sorted(frame_output_dir.glob("keyframe_*.jpg"))[:num_frames]
        extracted_paths = [str(f) for f in extracted_files]
        
        print(f"Extracted {len(extracted_paths)} key frames")
        return extracted_paths


class HybridVideoProcessor:
    """
    Hybrid video processor that can use both OpenCV and FFmpeg.
    Automatically chooses the best method based on the task.
    """
    
    def __init__(self, output_dir: str = "output/frames", prefer_ffmpeg: bool = True):
        """
        Initialize the hybrid processor.
        
        Args:
            output_dir: Directory to save extracted frames
            prefer_ffmpeg: Whether to prefer FFmpeg when available
        """
        self.output_dir = output_dir
        self.prefer_ffmpeg = prefer_ffmpeg
        
        # Initialize available processors
        self.opencv_processor = VideoFrameExtractor(output_dir)
        
        try:
            self.ffmpeg_processor = FFmpegExtractor(output_dir)
            self.has_ffmpeg = True
        except RuntimeError:
            self.ffmpeg_processor = None
            self.has_ffmpeg = False
            print("FFmpeg not available, using OpenCV only")
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video information using the best available method."""
        if self.has_ffmpeg and self.prefer_ffmpeg:
            try:
                return self.ffmpeg_processor.get_video_info(video_path)
            except Exception:
                return self.opencv_processor.get_video_info(video_path)
        else:
            return self.opencv_processor.get_video_info(video_path)
    
    def extract_frames(self, 
                      video_path: str, 
                      frame_interval: int = 1,
                      start_frame: int = 0,
                      end_frame: Optional[int] = None,
                      resize: Optional[Tuple[int, int]] = None,
                      quality: int = 95,
                      method: str = "auto") -> List[str]:
        """
        Extract frames using the best available method.
        
        Args:
            video_path: Path to the input video file
            frame_interval: Extract every nth frame
            start_frame: Frame number to start extraction
            end_frame: Frame number to end extraction
            resize: Tuple of (width, height) to resize frames
            quality: JPEG quality for saved frames
            method: "auto", "opencv", or "ffmpeg"
            
        Returns:
            List of paths to extracted frame files
        """
        # Determine which method to use
        use_ffmpeg = False
        
        if method == "ffmpeg":
            if not self.has_ffmpeg:
                raise RuntimeError("FFmpeg not available")
            use_ffmpeg = True
        elif method == "opencv":
            use_ffmpeg = False
        else:  # auto
            # Use FFmpeg for bulk extraction (high frame intervals)
            # Use OpenCV for precise frame selection or small intervals
            if self.has_ffmpeg and self.prefer_ffmpeg and frame_interval >= 10:
                use_ffmpeg = True
        
        if use_ffmpeg:
            print("Using FFmpeg for extraction")
            # Convert frame numbers to time for FFmpeg
            video_info = self.get_video_info(video_path)
            fps = video_info['fps']
            
            start_time = start_frame / fps if start_frame > 0 else None
            end_time = end_frame / fps if end_frame else None
            
            return self.ffmpeg_processor.extract_frames(
                video_path, frame_interval, start_time, end_time, resize, quality
            )
        else:
            print("Using OpenCV for extraction")
            return self.opencv_processor.extract_frames(
                video_path, frame_interval, start_frame, end_frame, resize, quality
            )
    
    def extract_key_frames(self, 
                          video_path: str, 
                          num_frames: int = 50,
                          method: str = "auto") -> List[str]:
        """
        Extract key frames using the best available method.
        
        Args:
            video_path: Path to the input video file
            num_frames: Number of frames to extract
            method: "auto", "opencv", "ffmpeg", "uniform", or "adaptive"
            
        Returns:
            List of paths to extracted frame files
        """
        if method == "ffmpeg" and self.has_ffmpeg:
            print("Using FFmpeg scene detection for key frames")
            return self.ffmpeg_processor.extract_key_frames(video_path, num_frames)
        elif method in ["opencv", "uniform", "adaptive"]:
            print(f"Using OpenCV {method} method for key frames")
            sampling_method = "uniform" if method == "opencv" else method
            return self.opencv_processor.extract_key_frames(video_path, num_frames, sampling_method)
        else:  # auto
            if self.has_ffmpeg and num_frames <= 100:
                # Use FFmpeg scene detection for smaller numbers of key frames
                try:
                    print("Using FFmpeg scene detection for key frames")
                    return self.ffmpeg_processor.extract_key_frames(video_path, num_frames)
                except Exception:
                    print("FFmpeg scene detection failed, falling back to OpenCV")
                    return self.opencv_processor.extract_key_frames(video_path, num_frames, "uniform")
            else:
                print("Using OpenCV uniform sampling for key frames")
                return self.opencv_processor.extract_key_frames(video_path, num_frames, "uniform")
    
    def benchmark_methods(self, video_path: str, frame_interval: int = 30) -> dict:
        """
        Benchmark both extraction methods if available.
        
        Args:
            video_path: Path to the input video file
            frame_interval: Frame interval for testing
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        results = {}
        
        # Test OpenCV
        print("Benchmarking OpenCV...")
        start_time = time.time()
        try:
            opencv_frames = self.opencv_processor.extract_frames(
                video_path, frame_interval, end_frame=300  # Limit for benchmark
            )
            opencv_time = time.time() - start_time
            results['opencv'] = {
                'time': opencv_time,
                'frames': len(opencv_frames),
                'fps': len(opencv_frames) / opencv_time
            }
        except Exception as e:
            results['opencv'] = {'error': str(e)}
        
        # Test FFmpeg if available
        if self.has_ffmpeg:
            print("Benchmarking FFmpeg...")
            start_time = time.time()
            try:
                # Clean previous extraction
                self.opencv_processor.clean_output_dir(Path(video_path).stem)
                
                ffmpeg_frames = self.ffmpeg_processor.extract_frames(
                    video_path, frame_interval, end_time=5.0  # 5 seconds for benchmark
                )
                ffmpeg_time = time.time() - start_time
                results['ffmpeg'] = {
                    'time': ffmpeg_time,
                    'frames': len(ffmpeg_frames),
                    'fps': len(ffmpeg_frames) / ffmpeg_time
                }
            except Exception as e:
                results['ffmpeg'] = {'error': str(e)}
        
        return results


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
    parser.add_argument("--ffmpeg", action="store_true",
                       help="Use FFmpeg for extraction (if available)")
    
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
