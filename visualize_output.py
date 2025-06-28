#!/usr/bin/env python3
"""
Quick visualization of the generated Gaussian reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_gaussians():
    """Create a 3D visualization of the generated Gaussians"""
    
    # Load the data
    positions = np.load('output/gaussian_reconstruction/standard_quality/positions.npy')
    colors = np.load('output/gaussian_reconstruction/standard_quality/colors.npy')
    
    print(f"Loaded {len(positions)} Gaussians")
    
    # Sample a subset for visualization (too many points would be slow)
    sample_size = min(5000, len(positions))
    indices = np.random.choice(len(positions), sample_size, replace=False)
    sample_positions = positions[indices]
    sample_colors = colors[indices]
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    
    # First subplot: 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    scatter = ax1.scatter(sample_positions[:, 0], 
                         sample_positions[:, 1], 
                         sample_positions[:, 2],
                         c=sample_colors,
                         s=0.5,
                         alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (Depth)')
    ax1.set_title(f'3D Gaussian Reconstruction\n({sample_size} of {len(positions)} points)')
    
    # Second subplot: Top view (X-Z)
    ax2 = fig.add_subplot(222)
    ax2.scatter(sample_positions[:, 0], sample_positions[:, 2], 
                c=sample_colors, s=0.5, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z (Depth)')
    ax2.set_title('Top View (X-Z)')
    ax2.grid(True)
    
    # Third subplot: Side view (Y-Z)
    ax3 = fig.add_subplot(223)
    ax3.scatter(sample_positions[:, 1], sample_positions[:, 2], 
                c=sample_colors, s=0.5, alpha=0.6)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z (Depth)')
    ax3.set_title('Side View (Y-Z)')
    ax3.grid(True)
    
    # Fourth subplot: Front view (X-Y)
    ax4 = fig.add_subplot(224)
    ax4.scatter(sample_positions[:, 0], sample_positions[:, 1], 
                c=sample_colors, s=0.5, alpha=0.6)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('Front View (X-Y)')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'output/gaussian_reconstruction/standard_quality/visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    # Display statistics
    print("\n" + "="*60)
    print("GAUSSIAN RECONSTRUCTION ANALYSIS")
    print("="*60)
    print(f"Total Gaussians: {len(positions):,}")
    print(f"Visualized sample: {sample_size:,}")
    print(f"\nSpatial extent:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] (range: {positions[:, 0].max() - positions[:, 0].min():.3f})")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] (range: {positions[:, 1].max() - positions[:, 1].min():.3f})")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] (range: {positions[:, 2].max() - positions[:, 2].min():.3f})")
    
    print(f"\nColor statistics:")
    print(f"  R channel: [{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}]")
    print(f"  G channel: [{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}]")
    print(f"  B channel: [{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")
    
    # Density analysis
    print(f"\nDensity analysis:")
    print(f"  Points per unit volume: {len(positions) / (np.prod([positions[:, i].max() - positions[:, i].min() for i in range(3)])):.2f}")
    
    plt.show()

if __name__ == "__main__":
    visualize_gaussians()
