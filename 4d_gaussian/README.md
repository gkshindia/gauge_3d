# 4D Gaussian Splatting Pipeline

This module implements 4D Gaussian Splatting for dynamic scene reconstruction from RGB-D video sequences.

## Overview

The 4D Gaussian Splatting pipeline converts RGB frames and depth maps from Phase 1 into a dynamic 3D representation using Gaussian primitives. This enables high-quality novel view synthesis and temporal consistency across the video sequence.

## Pipeline Stages

### Stage 1: Environment Setup
- **File**: `setup/environment_setup.py`
- **Purpose**: Install and configure all dependencies for 4D Gaussian Splatting
- **Key Dependencies**:
  - PyTorch3D for 3D operations
  - Differential Gaussian Rasterization
  - Open3D for point cloud processing
  - Simple-KNN for nearest neighbor operations

### Stage 2: Data Preparation
- **File**: `data_preparation/data_converter.py`
- **Purpose**: Convert RGB frames and depth maps to point clouds and camera poses
- **Features**:
  - Depth map to point cloud conversion
  - Camera pose estimation (COLMAP integration or simple trajectory)
  - Point cloud filtering and preprocessing
  - Temporal correspondence establishment

### Stage 3: Gaussian Initialization
- **File**: `gaussian_generation/gaussian_initializer.py`
- **Purpose**: Initialize 3D Gaussians from point cloud data
- **Features**:
  - Point cloud to Gaussian conversion
  - Temporal Gaussian linking
  - Adaptive scale estimation based on local density
  - Normal estimation for proper orientation

### Stage 4: Temporal Optimization (Coming Soon)
- **File**: `optimization/temporal_optimizer.py`
- **Purpose**: Optimize Gaussians across temporal windows
- **Features** (Planned):
  - Temporal consistency losses
  - Motion estimation and tracking
  - Adaptive Gaussian density control
  - Progressive training strategies

### Stage 5: Rendering (Coming Soon)
- **File**: `rendering/gaussian_renderer.py`
- **Purpose**: Real-time rendering and novel view synthesis
- **Features** (Planned):
  - Differentiable Gaussian rasterization
  - Temporal interpolation
  - Real-time preview capabilities
  - High-quality final rendering

## Quick Start

### 1. Environment Setup

First, ensure you have completed the depth estimation phase and have the required outputs:
- RGB frames in `output/frames/`
- Depth maps in `output/depth_maps/`
- Depth statistics in `output/depth_stats/`

### 2. Install 4D Gaussian Dependencies

```bash
# Run the setup script
cd 4d_gaussian
python setup/environment_setup.py
```

### 3. Run Complete 4D Gaussian Pipeline

```bash
# Run the complete pipeline for a video
python run_4d_gaussian.py 1080_60_fps

# With custom options
python run_4d_gaussian.py 1080_60_fps \
    --max-gaussians 500000 \
    --temporal-window 15 \
    --use-colmap \
    --output-dir output/my_gaussian_reconstruction
```

### 4. Setup Only

```bash
# Just run environment setup
python run_4d_gaussian.py --setup-only 1080_60_fps
```

## Configuration

### Gaussian Configuration
Edit `config/gaussian_config.py` to customize:

```python
@dataclass
class GaussianConfig:
    # Model settings
    device: str = "cuda"
    precision: str = "fp16"
    
    # Gaussian parameters
    initial_opacity: float = 0.1
    initial_scale: float = 0.01
    max_gaussians: int = 1000000
    
    # Temporal settings
    temporal_window: int = 10
    temporal_consistency_weight: float = 0.1
    
    # Quality settings
    depth_threshold: float = 0.01
    color_threshold: float = 0.05
```

## Input/Output Structure

### Input (from depth estimation)
```
output/
├── frames/
│   └── video_name/
│       ├── frame_000000.jpg
│       ├── frame_000006.jpg
│       └── ...
├── depth_maps/
│   └── video_name/
│       ├── frame_000000_depth.npy
│       ├── frame_000006_depth.npy
│       └── ...
└── depth_stats/
    └── video_name_depth_results.json
```

### Output (4D Gaussian)
```
output/gaussian_reconstruction/
├── point_clouds/
│   └── video_name/
│       ├── frame_000000_pointcloud.npy
│       └── ...
├── gaussian_init/
│   └── video_name/
│       ├── frame_000000_gaussians.npz
│       ├── temporal_correspondences.json
│       └── ...
├── logs/
│   └── 4d_gaussian_*.log
└── video_name_gaussian_data.json
```

## Key Features

### 1. Robust Point Cloud Generation
- Converts depth maps to colored point clouds
- Handles invalid depth values and outliers
- Applies bilateral filtering for noise reduction
- Statistical outlier removal using Open3D

### 2. Camera Pose Estimation
- COLMAP integration for structure-from-motion
- Fallback to simple circular trajectory
- Support for external camera parameters
- Intrinsic parameter estimation

### 3. Adaptive Gaussian Initialization
- Scale estimation based on local point density
- Normal estimation for proper orientation
- Temporal correspondence establishment
- Memory-efficient processing for large sequences

### 4. Temporal Consistency
- Frame-to-frame Gaussian correspondences
- Motion-aware initialization
- Temporal window processing
- Correspondence validation and filtering

## Performance Considerations

### Memory Management
- Processes videos in temporal windows
- Configurable maximum Gaussian count
- Efficient point cloud streaming
- GPU memory optimization

### Quality vs Speed
- Adjustable point cloud filtering levels
- Configurable correspondence thresholds
- Temporal window size optimization
- Progressive processing strategies

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `max_gaussians` parameter
   - Decrease `temporal_window` size
   - Use `precision: "fp16"` in config

2. **Missing Dependencies**
   - Run environment setup: `python setup/environment_setup.py`
   - Check CUDA compatibility
   - Verify PyTorch3D installation

3. **Poor Point Cloud Quality**
   - Check depth map quality from Phase 1
   - Adjust filtering parameters
   - Verify camera intrinsics

4. **COLMAP Fails**
   - Ensure COLMAP is installed and in PATH
   - Check image quality and overlap
   - Fallback to simple trajectory will be used

### Performance Optimization

1. **GPU Utilization**
   - Use CUDA if available
   - Enable mixed precision training
   - Batch process when possible

2. **Memory Optimization**
   - Process in smaller temporal windows
   - Use point cloud subsampling
   - Clear unused tensors regularly

## Dependencies

### Core Dependencies
- `torch>=2.0.0` - PyTorch for deep learning operations
- `pytorch3d` - 3D computer vision operations
- `open3d>=0.17.0` - Point cloud processing
- `numpy>=1.24.0` - Numerical computations

### Gaussian Splatting Specific
- `diff-gaussian-rasterization` - Differentiable rasterization
- `simple-knn` - Fast nearest neighbor search
- `plyfile>=0.7.4` - PLY file I/O
- `roma>=1.2.0` - Rotation representations

### Optional but Recommended
- `trimesh>=3.21.0` - Mesh processing
- `pymeshlab>=2022.2` - Advanced mesh operations
- `kornia>=0.6.12` - Computer vision operations
- `psutil>=5.9.0` - System resource monitoring

## Next Steps

After completing Phase 2, you can proceed to:

1. **Phase 3**: Temporal optimization and refinement
2. **Phase 4**: Real-time rendering and novel view synthesis
3. **Phase 5**: Distance measurement and analysis tools

## Contributing

When adding new features:

1. Follow the modular structure
2. Add comprehensive logging
3. Include error handling and validation
4. Update configuration options
5. Add visualization tools
6. Include performance metrics

## License

This module is part of the Gauge 3D project and follows the same license terms.
