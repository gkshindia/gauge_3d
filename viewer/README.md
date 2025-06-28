# Gauge 3D Visualization Suite

A comprehensive visualization toolkit for analyzing 3D reconstruction pipeline outputs including depth maps, point clouds, 4D Gaussian splats, and processing statistics.

## Overview

The visualization suite consists of five main modules:

- **DepthViewer**: Visualize and analyze depth estimation results
- **GaussianViewer**: Visualize and analyze 4D Gaussian reconstruction 
- **PointCloudViewer**: Visualize and analyze point clouds
- **PipelineViewer**: Overview of entire pipeline status and flow
- **StatsViewer**: Comprehensive statistics and quality metrics

## Installation

The visualization suite is already installed as part of the Gauge 3D project. Required dependencies:

```bash
# Core dependencies (already in pyproject.toml)
matplotlib
numpy
opencv-python
plotly  # For interactive visualizations
seaborn  # For enhanced statistical plots
pandas  # For data analysis

# Optional dependencies for enhanced 3D visualization
open3d  # For advanced 3D point cloud visualization
networkx  # For pipeline flow diagrams
```

## Quick Start

### Running the Demo

```bash
# Run the comprehensive demo showing all visualization capabilities
python demo_visualization.py
```

### Basic Usage Examples

```python
from viewer import DepthViewer, GaussianViewer, PointCloudViewer, PipelineViewer, StatsViewer

# Pipeline Overview
pipeline_viewer = PipelineViewer()
fig = pipeline_viewer.create_pipeline_overview()
plt.show()

# Depth Analysis
depth_viewer = DepthViewer()
datasets = depth_viewer.find_depth_datasets()
depth_stats = depth_viewer.load_depth_statistics(datasets[0])
fig = depth_viewer.plot_depth_statistics(depth_stats)
plt.show()

# Gaussian Analysis
gaussian_viewer = GaussianViewer()
gaussian_files = gaussian_viewer.find_gaussian_files()
gaussian_data = gaussian_viewer.load_gaussian_frame(gaussian_files[0])
fig = gaussian_viewer.visualize_gaussian_frame(gaussian_data)
plt.show()
```

## Module Documentation

### DepthViewer

Visualizes depth estimation outputs including depth maps, statistics, and temporal evolution.

**Key Methods:**
- `find_depth_datasets()`: Discover available depth datasets
- `load_depth_map(file_path)`: Load a single depth map
- `visualize_depth_sequence(files, ...)`: Visualize multiple depth maps
- `create_depth_video(...)`: Generate depth evolution video
- `plot_depth_statistics(stats)`: Visualize depth statistics

**Example:**
```python
depth_viewer = DepthViewer()

# Find and load depth data
datasets = depth_viewer.find_depth_datasets() 
depth_files = depth_viewer.find_depth_maps(datasets[0])

# Visualize depth sequence
fig = depth_viewer.visualize_depth_sequence(depth_files[:5])
plt.show()

# Create depth video
depth_viewer.create_depth_video(depth_files, "depth_evolution.mp4")
```

### GaussianViewer

Visualizes 4D Gaussian splat reconstructions with temporal analysis capabilities.

**Key Methods:**
- `find_gaussian_files()`: Find available Gaussian files
- `load_gaussian_frame(file_path)`: Load single frame Gaussians
- `visualize_gaussian_frame(data, ...)`: Visualize single frame
- `analyze_temporal_evolution(frames)`: Analyze changes over time
- `create_gaussian_animation(...)`: Generate Gaussian evolution video

**Example:**
```python
gaussian_viewer = GaussianViewer()

# Load Gaussian data
files = gaussian_viewer.find_gaussian_files()
frame_data = gaussian_viewer.load_gaussian_frame(files[0])

# Visualize
fig = gaussian_viewer.visualize_gaussian_frame(frame_data, plot_type='scatter')
plt.show()

# Temporal analysis
frames = [gaussian_viewer.load_gaussian_frame(f) for f in files[:10]]
stats = gaussian_viewer.analyze_temporal_evolution(frames)
```

### PointCloudViewer

Handles point cloud visualization with support for multiple formats (.ply, .pcd, .npz, .json).

**Key Methods:**
- `load_point_cloud(filepath)`: Load point cloud from various formats
- `visualize_point_cloud_matplotlib(points, colors)`: Matplotlib 3D visualization
- `visualize_point_cloud_plotly(points, colors)`: Interactive Plotly visualization  
- `visualize_point_cloud_open3d(points, colors)`: Open3D visualization
- `analyze_point_cloud_statistics(points)`: Compute statistics
- `compare_point_clouds(cloud_paths)`: Side-by-side comparison

**Example:**
```python
pc_viewer = PointCloudViewer()

# Load and visualize
cloud_data = pc_viewer.load_point_cloud("output/point_clouds/frame_001.ply")
fig = pc_viewer.visualize_point_cloud_matplotlib(
    cloud_data['points'], 
    cloud_data['colors']
)
plt.show()

# Interactive visualization (if Plotly available)
plotly_fig = pc_viewer.visualize_point_cloud_plotly(
    cloud_data['points'], 
    cloud_data['colors']
)
plotly_fig.show()
```

### PipelineViewer

Provides comprehensive overview of the entire 3D reconstruction pipeline.

**Key Methods:**
- `scan_pipeline_outputs()`: Analyze all pipeline stage outputs
- `create_pipeline_overview()`: Comprehensive pipeline visualization
- `generate_pipeline_report()`: Text-based pipeline status report
- `create_interactive_pipeline_dashboard()`: Interactive Plotly dashboard
- `compare_processing_runs(dirs)`: Compare multiple pipeline runs

**Example:**
```python
pipeline_viewer = PipelineViewer()

# Comprehensive overview
fig = pipeline_viewer.create_pipeline_overview()
plt.show()

# Generate detailed report
report = pipeline_viewer.generate_pipeline_report("pipeline_report.txt")
print(report)

# Interactive dashboard (if Plotly available)
dashboard = pipeline_viewer.create_interactive_pipeline_dashboard()
dashboard.show()
```

### StatsViewer

Analyzes and visualizes comprehensive statistics and quality metrics.

**Key Methods:**
- `load_depth_statistics(dataset)`: Load depth processing statistics
- `load_gaussian_statistics(dataset)`: Load Gaussian reconstruction statistics
- `plot_depth_statistics(stats)`: Comprehensive depth analysis plots
- `plot_gaussian_statistics(stats)`: Comprehensive Gaussian analysis plots
- `create_comprehensive_report()`: Full pipeline statistics report
- `compare_quality_across_datasets(patterns)`: Cross-dataset quality comparison

**Example:**
```python
stats_viewer = StatsViewer()

# Load and analyze depth statistics
depth_stats = stats_viewer.load_depth_statistics()
fig = stats_viewer.plot_depth_statistics(depth_stats)
plt.show()

# Load and analyze Gaussian statistics  
gaussian_stats = stats_viewer.load_gaussian_statistics()
fig = stats_viewer.plot_gaussian_statistics(gaussian_stats)
plt.show()

# Comprehensive report
report = stats_viewer.create_comprehensive_report("full_stats_report.txt")
```

## Output Directory Structure

The visualization tools expect the following output directory structure:

```
output/
├── frames/                    # Extracted video frames
│   └── {dataset}/
├── depth_maps/               # Depth estimation outputs
│   └── {dataset}/
├── depth_previews/           # Depth visualization previews
│   └── {dataset}/
├── depth_stats/              # Depth processing statistics
│   └── {dataset}_depth_results.json
├── gaussian_reconstruction/   # 4D Gaussian outputs
│   ├── {dataset}_gaussian_data.json
│   ├── point_clouds/
│   ├── logs/
│   └── gaussian_init/
└── test_output/              # Test outputs
```

## Advanced Usage

### Custom Visualization Settings

```python
# Depth visualization with custom settings
depth_viewer = DepthViewer()
fig = depth_viewer.visualize_depth_map(
    "output/depth_maps/1080_60_fps/frame_000001_depth.npy",
    colormap='plasma',
    title="Custom Depth Visualization",
    figsize=(15, 10)
)

# Gaussian visualization with specific plot type
gaussian_viewer = GaussianViewer()
fig = gaussian_viewer.visualize_gaussian_frame(
    gaussian_data,
    plot_type='heatmap',  # or 'scatter', 'density'
    color_by='size'       # or 'opacity', 'position'
)
```

### Batch Processing

```python
# Process all depth datasets
depth_viewer = DepthViewer()
datasets = depth_viewer.find_depth_datasets()

for dataset in datasets:
    stats = depth_viewer.load_depth_statistics(dataset)
    if stats:
        fig = depth_viewer.plot_depth_statistics(stats)
        plt.savefig(f"depth_analysis_{dataset}.png")
        plt.close()
```

### Export and Sharing

```python
# Export point cloud to different formats
pc_viewer = PointCloudViewer()
points, colors = load_your_point_cloud()

# Export to PLY
pc_viewer.export_point_cloud(points, colors, "output.ply")

# Export to NPZ
pc_viewer.export_point_cloud(points, colors, "output.npz")
```

### Interactive Dashboards

```python
# Create interactive pipeline dashboard
pipeline_viewer = PipelineViewer()
dashboard = pipeline_viewer.create_interactive_pipeline_dashboard()

# Save as HTML
dashboard.write_html("pipeline_dashboard.html")

# Or show in browser
dashboard.show()
```

## Troubleshooting

### Common Issues

1. **No data found**: Ensure the pipeline has been run and output files exist
2. **Visualization errors**: Check that required dependencies are installed
3. **Performance issues**: For large datasets, use sampling or reduce visualization resolution

### Dependencies Issues

```bash
# Install optional dependencies for enhanced features
pip install open3d plotly seaborn networkx

# For video creation
pip install ffmpeg-python
```

### Memory Management

For large datasets:

```python
# Use sampling for large point clouds
if len(points) > 100000:
    sample_indices = np.random.choice(len(points), 10000, replace=False)
    points = points[sample_indices]
    colors = colors[sample_indices] if colors is not None else None
```

## Integration with Pipeline

The visualization tools integrate seamlessly with the main pipeline:

```python
# After running depth estimation
from depth_estimation.main_pipeline import run_depth_pipeline
run_depth_pipeline("input_video.mp4", "1080_60_fps")

# Immediately visualize results
from viewer import DepthViewer
depth_viewer = DepthViewer()
stats = depth_viewer.load_depth_statistics("1080_60_fps")
fig = depth_viewer.plot_depth_statistics(stats)
plt.show()
```

## Contributing

To add new visualization features:

1. Create new methods in the appropriate viewer class
2. Follow the existing naming conventions
3. Add comprehensive docstrings
4. Include error handling and logging
5. Update the demo script with examples

## API Reference

For complete API documentation, see the individual module docstrings and method documentation within each viewer class.
