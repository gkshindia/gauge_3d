#!/usr/bin/env python3
"""
Test 4D Gaussian pipeline with different quality settings
"""

import subprocess
import time
import json
from pathlib import Path

def test_quality_preset(quality: str, max_frames: int = 5):
    """Test a quality preset with a limited number of frames"""
    print(f"\n{'='*60}")
    print(f"Testing Quality Preset: {quality}")
    print(f"{'='*60}")
    
    # Time the execution
    start_time = time.time()
    
    try:
        # Run the pipeline with quality preset
        cmd = [
            "python", "4d_gaussian/run_4d_gaussian.py", "1080_60_fps",
            "--quality", quality,
            "--skip-setup",
            "--temporal-window", "3"  # Smaller window for faster processing
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Success! Duration: {duration:.1f}s")
            
            # Check output file for statistics
            output_file = f"output/gaussian_reconstruction/1080_60_fps_gaussian_data.json"
            if Path(output_file).exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                print(f"📊 Results:")
                print(f"  - Frames processed: {data['num_frames']}")
                
                # Get point counts
                point_counts = [frame['point_count'] for frame in data['frames']]
                if point_counts:
                    avg_points = sum(point_counts) / len(point_counts)
                    print(f"  - Avg points per frame: {avg_points:,.0f}")
                    print(f"  - Point count range: {min(point_counts):,} - {max(point_counts):,}")
        else:
            print(f"❌ Failed! Duration: {duration:.1f}s")
            print(f"Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    """Test different quality presets"""
    print("4D Gaussian Pipeline Quality Testing")
    print(f"Testing on video: 1080_60_fps")
    
    # Test quality presets in order of increasing complexity
    qualities = ["preview", "standard"]  # Start with faster ones
    
    for quality in qualities:
        test_quality_preset(quality)
        
        # Give user a chance to stop
        try:
            input("\\nPress Enter to continue to next quality preset (or Ctrl+C to stop)...")
        except KeyboardInterrupt:
            print("\\nStopping tests.")
            break
    
    print("\\n🎉 Quality testing complete!")
    print("\\nRecommendations:")
    print("- 'preview': Use for quick iterations and testing")
    print("- 'standard': Best balance for most applications") 
    print("- 'high': Use when quality is more important than speed")
    print("- 'ultra': Use only for final high-quality reconstructions")

if __name__ == "__main__":
    main()
