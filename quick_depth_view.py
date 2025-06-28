#!/usr/bin/env python3
"""
Quick launch script for depth visualization
Usage: python quick_depth_view.py [viewer_type] [dataset_name] [start_frame]

viewer_type: tkinter (default), matplotlib, static, stats, enhanced, video, compare
dataset_name: auto-detect if not specified
start_frame: 0 if not specified
"""

import sys
import argparse
from pathlib import Path

# Add the viewer module to the path
sys.path.append(str(Path(__file__).parent))

from viewer.depth_viewer import DepthViewer

def main():
    parser = argparse.ArgumentParser(description='Quick depth viewer launcher')
    parser.add_argument('viewer_type', nargs='?', default='tkinter',
                       choices=['tkinter', 'matplotlib', 'static', 'stats', 'enhanced', 'video', 'compare'],
                       help='Type of viewer to launch')
    parser.add_argument('dataset_name', nargs='?', default=None,
                       help='Dataset name (auto-detect if not specified)')
    parser.add_argument('start_frame', nargs='?', type=int, default=0,
                       help='Starting frame number')
    
    args = parser.parse_args()
    
    # Initialize viewer
    viewer = DepthViewer()
    datasets = viewer.get_available_datasets()
    
    if not datasets:
        print("❌ No depth datasets found!")
        return
    
    # Select dataset
    if args.dataset_name and args.dataset_name in datasets:
        dataset = args.dataset_name
    else:
        dataset = datasets[0]
        if args.dataset_name:
            print(f"⚠️  Dataset '{args.dataset_name}' not found, using '{dataset}'")
    
    frame_count = viewer.get_frame_count(dataset)
    print(f"🎬 Loading dataset '{dataset}' ({frame_count} frames)")
    
    # Validate start frame
    start_frame = max(0, min(args.start_frame, frame_count - 1))
    if start_frame != args.start_frame:
        print(f"⚠️  Start frame adjusted to {start_frame}")
    
    # Launch appropriate viewer
    try:
        if args.viewer_type == 'tkinter':
            print("🚀 Launching Tkinter GUI viewer...")
            viewer.create_tkinter_viewer(dataset, start_frame)
            
        elif args.viewer_type == 'matplotlib':
            print("🚀 Launching matplotlib interactive viewer...")
            viewer.interactive_frame_viewer(dataset, start_frame)
            
        elif args.viewer_type == 'static':
            print("📊 Showing static sequence...")
            import matplotlib.pyplot as plt
            fig = viewer.visualize_depth_sequence(dataset, start_frame, num_frames=5)
            plt.show()
            
        elif args.viewer_type == 'stats':
            print("📈 Showing basic statistics...")
            import matplotlib.pyplot as plt
            fig = viewer.visualize_depth_statistics(dataset)
            if fig:
                plt.show()
            else:
                print("❌ No statistics available")
                
        elif args.viewer_type == 'enhanced':
            print("📈 Showing enhanced statistics...")
            import matplotlib.pyplot as plt
            fig = viewer.visualize_depth_statistics_enhanced(dataset)
            if fig:
                plt.show()
            else:
                print("❌ No statistics available")
                
        elif args.viewer_type == 'video':
            print("🎥 Creating depth video...")
            video_path = viewer.create_depth_video(dataset, fps=30)
            print(f"✅ Video saved to: {video_path}")
            
        elif args.viewer_type == 'compare':
            print("🔄 Comparing frame sequences...")
            import matplotlib.pyplot as plt
            early_frames = [start_frame + i*6 for i in range(5)]
            later_frames = [min(frame_count-1, start_frame + 100 + i*6) for i in range(5)]
            
            fig = viewer.compare_depth_sequences(dataset, 
                                               [early_frames, later_frames], 
                                               ["Early Sequence", "Later Sequence"])
            plt.show()
            
    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
