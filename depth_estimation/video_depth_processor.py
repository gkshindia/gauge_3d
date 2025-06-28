"""
Video Depth Processor for DepthAnyVideo (DA-V) Integration
Step 1.2: Video Preprocessing with DA-V Integration
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import logging
from dataclasses import dataclass

# Add parent src directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import video processor components
try:
    from video_processor import VideoFrameExtractor
except ImportError:
    # Fallback if video_processor is not available
    logger = logging.getLogger(__name__)
    logger.warning("video_processor module not found, using minimal implementation")
    
    class VideoFrameExtractor:
        def __init__(self, output_dir: str):
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        def get_video_info(self, video_path: str) -> dict:
            import cv2
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline."""
    # Frame extraction
    frame_interval: int = 1
    max_frames: Optional[int] = None
    target_fps: Optional[float] = None
    resize_target: Optional[Tuple[int, int]] = None
    
    # Quality assessment
    blur_threshold: float = 100.0  # Laplacian variance threshold
    brightness_range: Tuple[float, float] = (20.0, 235.0)  # Min/max brightness
    motion_threshold: float = 50.0  # Motion detection threshold
    
    # Video stabilization
    enable_stabilization: bool = False
    stabilization_method: str = "optical_flow"  # or "feature_based"
    
    # DA-V specific
    depth_model_size: str = "base"  # "base" or "large"
    batch_size: int = 4
    device: str = "auto"  # "auto", "cuda", "cpu"


class FrameQualityAssessor:
    """Assess frame quality for depth estimation."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def assess_blur(self, frame: np.ndarray) -> float:
        """
        Assess frame blur using Laplacian variance.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Blur score (higher = sharper)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def assess_brightness(self, frame: np.ndarray) -> float:
        """
        Assess frame brightness.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Average brightness value
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def assess_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Assess motion between consecutive frames.
        
        Args:
            frame1: Previous frame (BGR)
            frame2: Current frame (BGR)
            
        Returns:
            Motion score
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, 
            np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
            None
        )[0]
        
        # Calculate motion magnitude
        if flow is not None and len(flow) > 0:
            motion = np.sqrt(flow[0][0][0]**2 + flow[0][0][1]**2)
            return float(motion)
        return 0.0
    
    def is_frame_acceptable(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> Dict[str, Union[bool, float]]:
        """
        Determine if frame meets quality criteria.
        
        Args:
            frame: Current frame to assess
            prev_frame: Previous frame for motion assessment
            
        Returns:
            Dictionary with assessment results
        """
        results = {}
        
        # Blur assessment
        blur_score = self.assess_blur(frame)
        results['blur_score'] = blur_score
        results['blur_acceptable'] = blur_score >= self.config.blur_threshold
        
        # Brightness assessment
        brightness = self.assess_brightness(frame)
        results['brightness'] = brightness
        results['brightness_acceptable'] = (
            self.config.brightness_range[0] <= brightness <= self.config.brightness_range[1]
        )
        
        # Motion assessment (if previous frame provided)
        if prev_frame is not None:
            motion_score = self.assess_motion(prev_frame, frame)
            results['motion_score'] = motion_score
            results['motion_acceptable'] = motion_score <= self.config.motion_threshold
        else:
            results['motion_score'] = 0.0
            results['motion_acceptable'] = True
        
        # Overall acceptance
        results['overall_acceptable'] = (
            results['blur_acceptable'] and 
            results['brightness_acceptable'] and 
            results['motion_acceptable']
        )
        
        return results


class VideoStabilizer:
    """Video stabilization for improved depth estimation."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.prev_gray = None
        self.transforms = []
        
    def reset(self):
        """Reset stabilizer state."""
        self.prev_gray = None
        self.transforms = []
    
    def stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize a single frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Stabilized frame
        """
        if not self.config.enable_stabilization:
            return frame
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        
        if self.config.stabilization_method == "optical_flow":
            return self._stabilize_optical_flow(frame, gray)
        elif self.config.stabilization_method == "feature_based":
            return self._stabilize_feature_based(frame, gray)
        else:
            logger.warning(f"Unknown stabilization method: {self.config.stabilization_method}")
            return frame
    
    def _stabilize_optical_flow(self, frame: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Stabilize using optical flow."""
        # Detect corners in previous frame
        corners = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=10
        )
        
        if corners is not None and len(corners) > 10:
            # Calculate optical flow
            new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, corners, None
            )
            
            # Filter good points
            good_old = corners[status == 1]
            good_new = new_corners[status == 1]
            
            if len(good_old) > 10:
                # Estimate transformation
                transform = cv2.estimateAffinePartial2D(good_old, good_new)[0]
                
                if transform is not None:
                    # Apply stabilization
                    h, w = frame.shape[:2]
                    stabilized = cv2.warpAffine(frame, transform, (w, h))
                    self.prev_gray = gray
                    return stabilized
        
        self.prev_gray = gray
        return frame
    
    def _stabilize_feature_based(self, frame: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Stabilize using feature matching."""
        # Create ORB detector
        orb = cv2.ORB_create()
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        
        if des1 is not None and des2 is not None:
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) > 10:
                # Extract matched points
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Estimate transformation
                transform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
                
                if transform is not None:
                    # Apply stabilization
                    h, w = frame.shape[:2]
                    stabilized = cv2.warpAffine(frame, transform, (w, h))
                    self.prev_gray = gray
                    return stabilized
        
        self.prev_gray = gray
        return frame


class VideoDepthProcessor:
    """
    Enhanced video processor with DA-V integration for depth estimation.
    Step 1.2: Video Preprocessing with quality assessment and stabilization.
    """
    
    def __init__(self, config: ProcessingConfig, output_dir: str = "output"):
        """
        Initialize the video depth processor.
        
        Args:
            config: Processing configuration
            output_dir: Base output directory
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.quality_dir = self.output_dir / "quality_reports"
        
        # Create directories
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.frame_extractor = VideoFrameExtractor(str(self.frames_dir))
        self.quality_assessor = FrameQualityAssessor(config)
        self.stabilizer = VideoStabilizer(config)
        
        logger.info(f"Initialized VideoDepthProcessor with config: {config}")
    
    def preprocess_video(self, video_path: str) -> Dict[str, any]:
        """
        Preprocess video for depth estimation.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary with preprocessing results
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Starting video preprocessing: {video_path}")
        
        # Get video info
        video_info = self.frame_extractor.get_video_info(str(video_path))
        logger.info(f"Video info: {video_info}")
        
        # Calculate processing parameters
        total_frames = video_info['total_frames']
        fps = video_info['fps']
        
        # Determine frame selection strategy
        if self.config.target_fps:
            frame_interval = max(1, int(fps / self.config.target_fps))
        else:
            frame_interval = self.config.frame_interval
        
        if self.config.max_frames:
            estimated_frames = total_frames // frame_interval
            if estimated_frames > self.config.max_frames:
                frame_interval = total_frames // self.config.max_frames
        
        logger.info(f"Processing with frame interval: {frame_interval}")
        
        # Process video
        results = self._process_video_frames(str(video_path), frame_interval, video_info)
        
        # Save quality report
        self._save_quality_report(video_path.stem, results)
        
        logger.info("Video preprocessing completed")
        return results
    
    def _process_video_frames(self, video_path: str, frame_interval: int, video_info: Dict) -> Dict:
        """Process video frames with quality assessment and stabilization."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_name = Path(video_path).stem
        video_frames_dir = self.frames_dir / video_name
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset stabilizer
        self.stabilizer.reset()
        
        results = {
            'video_info': video_info,
            'processed_frames': [],
            'quality_stats': {
                'total_extracted': 0,
                'total_accepted': 0,
                'avg_blur_score': 0.0,
                'avg_brightness': 0.0,
                'avg_motion_score': 0.0
            },
            'frame_interval': frame_interval,
            'output_dir': str(video_frames_dir)
        }
        
        frame_count = 0
        saved_count = 0
        prev_frame = None
        quality_scores = []
        
        total_to_process = video_info['total_frames'] // frame_interval
        
        with tqdm(total=total_to_process, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % frame_interval == 0:
                    # Resize if specified
                    if self.config.resize_target:
                        frame = cv2.resize(frame, self.config.resize_target)
                    
                    # Assess quality
                    quality = self.quality_assessor.is_frame_acceptable(frame, prev_frame)
                    quality_scores.append(quality)
                    
                    # Stabilize frame
                    stabilized_frame = self.stabilizer.stabilize_frame(frame)
                    
                    # Save frame if acceptable or if quality filtering is disabled
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = video_frames_dir / frame_filename
                    
                    if quality['overall_acceptable'] or not hasattr(self.config, 'enable_quality_filter') or not self.config.enable_quality_filter:
                        cv2.imwrite(str(frame_path), stabilized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        results['processed_frames'].append({
                            'frame_number': frame_count,
                            'filename': frame_filename,
                            'path': str(frame_path),
                            'quality': quality,
                            'accepted': True
                        })
                        
                        results['quality_stats']['total_accepted'] += 1
                        saved_count += 1
                    else:
                        results['processed_frames'].append({
                            'frame_number': frame_count,
                            'filename': frame_filename,
                            'quality': quality,
                            'accepted': False
                        })
                    
                    results['quality_stats']['total_extracted'] += 1
                    prev_frame = frame
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        # Calculate quality statistics
        if quality_scores:
            results['quality_stats']['avg_blur_score'] = np.mean([q['blur_score'] for q in quality_scores])
            results['quality_stats']['avg_brightness'] = np.mean([q['brightness'] for q in quality_scores])
            results['quality_stats']['avg_motion_score'] = np.mean([q['motion_score'] for q in quality_scores])
        
        logger.info(f"Processed {results['quality_stats']['total_extracted']} frames, "
                   f"accepted {results['quality_stats']['total_accepted']}")
        
        return results
    
    def _save_quality_report(self, video_name: str, results: Dict):
        """Save quality assessment report."""
        report_path = self.quality_dir / f"{video_name}_quality_report.json"
        
        import json
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Quality report saved: {report_path}")
    
    def get_processed_frames(self, video_name: str) -> List[str]:
        """
        Get list of processed frame paths for a video.
        
        Args:
            video_name: Name of the video (without extension)
            
        Returns:
            List of frame file paths
        """
        video_frames_dir = self.frames_dir / video_name
        if not video_frames_dir.exists():
            return []
        
        frame_files = sorted(video_frames_dir.glob("frame_*.jpg"))
        return [str(f) for f in frame_files]


def create_default_config() -> ProcessingConfig:
    """Create default processing configuration."""
    return ProcessingConfig(
        frame_interval=1,
        max_frames=1000,  # Limit for memory management
        target_fps=None,
        resize_target=None,
        blur_threshold=100.0,
        brightness_range=(20.0, 235.0),
        motion_threshold=50.0,
        enable_stabilization=False,
        stabilization_method="optical_flow",
        depth_model_size="base",
        batch_size=4,
        device="auto"
    )


if __name__ == "__main__":
    """Example usage for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video preprocessing for depth estimation")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--frame-interval", type=int, default=1, help="Frame extraction interval")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--enable-stabilization", action="store_true", help="Enable video stabilization")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), help="Resize frames")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    config.frame_interval = args.frame_interval
    if args.max_frames:
        config.max_frames = args.max_frames
    config.enable_stabilization = args.enable_stabilization
    if args.resize:
        config.resize_target = tuple(args.resize)
    
    # Process video
    processor = VideoDepthProcessor(config, args.output_dir)
    results = processor.preprocess_video(args.video_path)
    
    print("\nPreprocessing Results:")
    print(f"Total frames extracted: {results['quality_stats']['total_extracted']}")
    print(f"Frames accepted: {results['quality_stats']['total_accepted']}")
    print(f"Average blur score: {results['quality_stats']['avg_blur_score']:.2f}")
    print(f"Average brightness: {results['quality_stats']['avg_brightness']:.2f}")
    print(f"Average motion score: {results['quality_stats']['avg_motion_score']:.2f}")
    print(f"Output directory: {results['output_dir']}")
