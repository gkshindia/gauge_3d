#!/usr/bin/env python3
"""
Create a properly paired test subset for 4D Gaussian pipeline testing
"""

import os
import shutil
from pathlib import Path
import numpy as np

def create_paired_test_subset():
    """Create a test subset with properly paired frames and depth maps"""
    
    # Source directories
    frames_dir = Path("output/frames/1080_60_fps")
    depth_maps_dir = Path("output/depth_maps/1080_60_fps")
    
    # Target directories
    test_frames_dir = Path("output/test_frames/1080_5_frames")
    test_depth_maps_dir = Path("output/test_depth_maps/1080_5_frames")
    
    # Create target directories
    test_frames_dir.mkdir(parents=True, exist_ok=True)
    test_depth_maps_dir.mkdir(parents=True, exist_ok=True)
    
    # Get first 5 depth files that have corresponding frames
    depth_files = sorted(list(depth_maps_dir.glob("*_depth.npy")))
    selected_pairs = []
    
    for depth_file in depth_files:
        # Extract frame number from depth filename
        depth_name = depth_file.name
        if depth_name.startswith("frame_") and depth_name.endswith("_depth.npy"):
            frame_num = depth_name[6:-10]  # Extract the number part
            corresponding_frame = frames_dir / f"frame_{frame_num}.jpg"
            
            if corresponding_frame.exists():
                selected_pairs.append((corresponding_frame, depth_file))
                
                if len(selected_pairs) >= 5:
                    break
    
    print(f"Selected {len(selected_pairs)} frame-depth pairs:")
    
    # Copy files to test directories
    for i, (frame_file, depth_file) in enumerate(selected_pairs):
        # Copy frame
        target_frame = test_frames_dir / frame_file.name
        shutil.copy2(frame_file, target_frame)
        print(f"  {frame_file.name} + {depth_file.name}")
        
        # Copy depth map
        target_depth = test_depth_maps_dir / depth_file.name
        shutil.copy2(depth_file, target_depth)
        
        # Check depth map
        depth = np.load(depth_file)
        print(f"    Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
    
    print(f"\\n✅ Properly paired test subset created with {len(selected_pairs)} frames")
    print(f"Frames: {test_frames_dir}")
    print(f"Depth maps: {test_depth_maps_dir}")
    
    return test_frames_dir.parent.name

if __name__ == "__main__":
    video_name = create_paired_test_subset()
    print(f"\\nTo test 4D Gaussian pipeline:")
    print(f"python 4d_gaussian/run_4d_gaussian.py {video_name} --frames-dir output/test_frames --depth-maps-dir output/test_depth_maps")
