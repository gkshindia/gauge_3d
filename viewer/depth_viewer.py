"""
Simple Depth Map Viewer with Tkinter interface.
Usage: python depth_viewer.py --dataset DATASET_NAME [--frame START_FRAME]
"""

import numpy as np
import matplotlib.cm as cm
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse

class DepthViewer:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.depth_maps_dir = self.output_dir / "depth_maps"
        
    def load_depth_map(self, dataset_name: str, frame_idx: int) -> np.ndarray:
        """Load a single depth map."""
        depth_file = self.depth_maps_dir / dataset_name / f"frame_{frame_idx:06d}_depth.npy"
        if depth_file.exists():
            return np.load(depth_file)
        return None
    
    def get_frame_count(self, dataset_name: str) -> int:
        """Get the number of frames in a dataset."""
        depth_dir = self.depth_maps_dir / dataset_name
        if not depth_dir.exists():
            return 0
        depth_files = list(depth_dir.glob("frame_*_depth.npy"))
        return len(depth_files)
    
    def get_available_datasets(self):
        """Get list of available depth datasets."""
        if not self.depth_maps_dir.exists():
            return []
        datasets = [d.name for d in self.depth_maps_dir.iterdir() if d.is_dir()]
        return sorted(datasets)
    
    def view_depth_maps(self, dataset_name: str, start_frame: int = 0):
        """Launch Tkinter viewer for depth maps."""
        total_frames = self.get_frame_count(dataset_name)
        if total_frames == 0:
            print(f"No frames found in dataset: {dataset_name}")
            return
        
        # Initialize Tkinter window
        self.root = tk.Tk()
        self.root.title(f"Depth Viewer - {dataset_name}")
        self.root.geometry("1200x800")
        
        # Initialize state
        self.current_frame = max(0, min(start_frame, total_frames - 1))
        self.total_frames = total_frames
        self.dataset_name = dataset_name
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display
        self.image_label = ttk.Label(main_frame)
        self.image_label.pack(pady=10)
        
        # Frame info
        self.info_var = tk.StringVar()
        info_label = ttk.Label(main_frame, textvariable=self.info_var, font=("Arial", 12))
        info_label.pack(pady=5)
        
        # Controls
        controls = ttk.Frame(main_frame)
        controls.pack(pady=10)
        
        ttk.Button(controls, text="← Previous", command=self._prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="Next →", command=self._next_frame).pack(side=tk.LEFT, padx=5)
        
        # Jump to frame
        jump_frame = ttk.Frame(main_frame)
        jump_frame.pack(pady=5)
        
        ttk.Label(jump_frame, text="Jump to:").pack(side=tk.LEFT, padx=5)
        self.jump_entry = ttk.Entry(jump_frame, width=10)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(jump_frame, text="Go", command=self._jump_to_frame).pack(side=tk.LEFT, padx=5)
        
        # Frame slider
        self.progress_var = tk.DoubleVar()
        self.slider = ttk.Scale(main_frame, from_=0, to=total_frames-1, 
                               orient=tk.HORIZONTAL, variable=self.progress_var,
                               command=self._on_slider_change)
        self.slider.pack(fill=tk.X, pady=10, padx=20)
        
        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self._prev_frame())
        self.root.bind('<Right>', lambda e: self._next_frame())
        self.root.bind('<Return>', lambda e: self._jump_to_frame())
        
        # Show first frame
        self._update_display()
        
        # Start GUI
        self.root.mainloop()
    
    def _update_display(self):
        """Update the display with current frame."""
        depth_map = self.load_depth_map(self.dataset_name, self.current_frame)
        
        if depth_map is not None:
            # Normalize depth map
            depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            # Apply colormap and convert to image
            colored_depth = cm.get_cmap('viridis')(depth_norm)
            colored_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)
            
            # Resize for display if needed
            h, w = colored_depth.shape[:2]
            max_size = 800
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                colored_depth = Image.fromarray(colored_depth).resize((new_w, new_h))
            else:
                colored_depth = Image.fromarray(colored_depth)
            
            # Convert to Tkinter image and display
            self.tk_image = ImageTk.PhotoImage(colored_depth)
            self.image_label.configure(image=self.tk_image)
            
            # Update info text
            info = (f"Frame {self.current_frame}/{self.total_frames-1} | "
                   f"Min: {depth_map.min():.3f} | Max: {depth_map.max():.3f} | "
                   f"Mean: {depth_map.mean():.3f}")
        else:
            info = f"Frame {self.current_frame}/{self.total_frames-1} - No data"
        
        self.info_var.set(info)
        self.progress_var.set(self.current_frame)
    
    def _next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self._update_display()
    
    def _prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self._update_display()
    
    def _jump_to_frame(self):
        try:
            frame = int(self.jump_entry.get())
            if 0 <= frame < self.total_frames:
                self.current_frame = frame
                self._update_display()
        except ValueError:
            pass
    
    def _on_slider_change(self, value):
        frame = int(float(value))
        if frame != self.current_frame:
            self.current_frame = frame
            self._update_display()

def main():
    parser = argparse.ArgumentParser(description="Depth Map Viewer")
    parser.add_argument("--dataset", type=str, help="Dataset name to view")
    parser.add_argument("--frame", type=int, default=0, help="Starting frame number")
    args = parser.parse_args()
    
    viewer = DepthViewer()
    datasets = viewer.get_available_datasets()
    
    if not datasets:
        print("No depth datasets found!")
        return
    
    if args.dataset and args.dataset in datasets:
        dataset = args.dataset
    else:
        dataset = datasets[0]
        if args.dataset:
            print(f"Dataset '{args.dataset}' not found, using '{dataset}'")
    
    viewer.view_depth_maps(dataset, args.frame)

if __name__ == "__main__":
    main()
