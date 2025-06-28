#!/usr/bin/env python3
"""
Test module for depth estimation functionality
"""

import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def test_depth_pipeline_import():
    """Test that depth estimation modules can be imported"""
    print("Testing depth pipeline imports...")
    
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        
        from depth_estimation.depth_pipeline import DepthEstimationPipeline
        print("✅ DepthEstimationPipeline import successful")
        
        from depth_estimation.video_depth_processor import VideoDepthProcessor  
        print("✅ VideoDepthProcessor import successful")
        
        from depth_estimation.output_manager import DepthOutputManager
        print("✅ DepthOutputManager import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Depth pipeline import failed: {e}")
        return False


def test_depth_config():
    """Test depth estimation configuration"""
    print("Testing depth configuration...")
    
    try:
        project_root = Path(__file__).parent.parent
        config_file = project_root / "depth_estimation" / "config" / "dav_config.yaml"
        
        if config_file.exists():
            print("✅ Depth config file exists")
            return True
        else:
            print("❌ Depth config file missing")
            return False
            
    except Exception as e:
        print(f"❌ Depth config test failed: {e}")
        return False


def test_depth_model_setup():
    """Test depth model setup (without actually loading models)"""
    print("Testing depth model setup...")
    
    try:
        # Test that we can import the required libraries
        import torch
        import transformers
        
        print("✅ PyTorch and Transformers available for depth models")
        
        # Test CUDA availability for depth processing
        if torch.cuda.is_available():
            print("✅ CUDA available for depth processing")
        else:
            print("⚠️  CUDA not available - depth processing will be slower")
        
        return True
        
    except ImportError as e:
        print(f"❌ Depth model setup failed: {e}")
        return False


def test_depth_output_paths():
    """Test depth estimation output path structure"""
    print("Testing depth output paths...")
    
    try:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "output"
        
        # Expected output subdirectories for depth processing
        expected_subdirs = [
            "depth_maps",
            "depth_previews",
            "depth_stats", 
            "frames"
        ]
        
        all_exist = True
        
        for subdir in expected_subdirs:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                print(f"✅ Output path exists: {subdir}/")
            else:
                print(f"⚠️  Output path will be created: {subdir}/")
        
        return True
        
    except Exception as e:
        print(f"❌ Depth output paths test failed: {e}")
        return False


def test_video_input():
    """Test video input validation"""
    print("Testing video input...")
    
    try:
        project_root = Path(__file__).parent.parent
        vinput_dir = project_root / "vinput"
        
        if not vinput_dir.exists():
            print("❌ Video input directory missing")
            return False
            
        # Look for video files
        video_files = list(vinput_dir.glob("*.mp4"))
        
        if video_files:
            print(f"✅ Found {len(video_files)} video file(s) for testing")
            for video_file in video_files:
                print(f"   - {video_file.name}")
            return True
        else:
            print("⚠️  No video files found in vinput/ directory")
            return True  # Not an error - videos may be added later
            
    except Exception as e:
        print(f"❌ Video input test failed: {e}")
        return False


def test_depth_viewer():
    """Test depth viewer functionality"""
    print("Testing depth viewer...")
    
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        
        from viewer.depth_viewer import DepthViewer
        print("✅ DepthViewer import successful")
        
        # Test that required dependencies are available
        import tkinter
        import matplotlib
        
        print("✅ GUI dependencies available for depth viewer")
        
        return True
        
    except ImportError as e:
        print(f"❌ Depth viewer test failed: {e}")
        return False
