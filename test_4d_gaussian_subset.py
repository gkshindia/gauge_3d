#!/usr/bin/env python3
"""
Test script for 4D Gaussian pipeline with limited frames
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np

def create_test_subset():
    """Create a test subset with just 5 frames"""
    
    # Create test directories
    test_frames_dir = Path("output/test_frames/1080_5_frames")
    test_depth_dir = Path("output/test_depth_maps/1080_5_frames")
    
    test_frames_dir.mkdir(parents=True, exist_ok=True)
    test_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy first 5 frames and depth maps
    source_frames = Path("output/frames/1080_60_fps")
    source_depth = Path("output/depth_maps/1080_60_fps")
    
    frame_files = sorted(list(source_frames.glob("frame_*.jpg")))[:5]
    depth_files = sorted(list(source_depth.glob("frame_*_depth.npy")))[:5]
    
    print(f"Copying {len(frame_files)} frames...")
    for frame_file in frame_files:
        shutil.copy2(frame_file, test_frames_dir / frame_file.name)
    
    print(f"Copying {len(depth_files)} depth maps...")
    for depth_file in depth_files:
        shutil.copy2(depth_file, test_depth_dir / depth_file.name)
    
    print(f"✅ Test subset created:")
    print(f"   Frames: {test_frames_dir}")
    print(f"   Depth maps: {test_depth_dir}")
    
    return test_frames_dir, test_depth_dir

if __name__ == "__main__":
    create_test_subset()
