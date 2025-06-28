#!/usr/bin/env python3
"""
Demo script to test the interactive depth viewer functionality
"""

import sys
from pathlib import Path

# Add the viewer module to the path
sys.path.append(str(Path(__file__).parent))

from viewer.depth_viewer import DepthViewer

def main():
    """Demo the depth viewer functionality."""
    print("=== Depth Viewer Demo ===")
    
    # Initialize the viewer
    viewer = DepthViewer()
    datasets = viewer.get_available_datasets()
    
    if not datasets:
        print("❌ No depth datasets found!")
        print("Expected datasets in: output/depth_maps/")
        return
    
    print(f"✅ Found {len(datasets)} dataset(s): {datasets}")
    
    # Use the first dataset
    dataset = datasets[0]
    frame_count = viewer.get_frame_count(dataset)
    print(f"📊 Dataset '{dataset}' has {frame_count} frames")
    
    if frame_count == 0:
        print("❌ No frames found in dataset!")
        return
    
    print("\n🎬 Choose visualization mode:")
    print("1. 🖱️  Tkinter GUI viewer (recommended - mouse + keyboard)")
    print("2. ⌨️  Matplotlib interactive viewer (keyboard only)")
    print("3. 📊 Static sequence view (5 frames)")
    print("4. 📈 Statistics view")
    print("5. 🎥 Create depth video")
    
    try:
        choice = input("\nEnter choice (1-5, default=1): ").strip() or "1"
        
        if choice == "1":
            print(f"\n🚀 Starting Tkinter GUI viewer for '{dataset}'...")
            print("Navigation:")
            print("  - Use arrow keys or buttons to navigate")
            print("  - Drag the slider to scrub through frames")
            print("  - Enter frame number and click 'Go' to jump")
            print("  - Close window when done")
            viewer.create_tkinter_viewer(dataset, start_frame=0)
            
        elif choice == "2":
            print(f"\n🚀 Starting matplotlib interactive viewer for '{dataset}'...")
            print("Navigation:")
            print("  - Left/Right arrows or 'p'/'n' keys to navigate")
            print("  - Press 'j' to jump to specific frame")
            print("  - Press 'q' to quit")
            viewer.interactive_frame_viewer(dataset, start_frame=0)
            
        elif choice == "3":
            print(f"\n📊 Showing static sequence for '{dataset}'...")
            import matplotlib.pyplot as plt
            fig = viewer.visualize_depth_sequence(dataset, start_frame=0, num_frames=5)
            plt.show()
            
        elif choice == "4":
            print(f"\n📈 Showing statistics for '{dataset}'...")
            import matplotlib.pyplot as plt
            fig = viewer.visualize_depth_statistics(dataset)
            if fig:
                plt.show()
            else:
                print("❌ No statistics file found")
                
        elif choice == "5":
            print(f"\n🎥 Creating depth video for '{dataset}'...")
            try:
                video_path = viewer.create_depth_video(dataset, fps=30)
                print(f"✅ Video created: {video_path}")
            except Exception as e:
                print(f"❌ Video creation failed: {e}")
                
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n🛑 Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔄 Falling back to static sequence view...")
        import matplotlib.pyplot as plt
        fig = viewer.visualize_depth_sequence(dataset, start_frame=0, num_frames=5)
        plt.show()

if __name__ == "__main__":
    main()
