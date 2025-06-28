# Gauge 3D: Complete Documentation and User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Pipeline Overview](#pipeline-overview)
5. [Usage Examples](#usage-examples)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [API Reference](#api-reference)
10. [Contributing](#contributing)

## Introduction

Gauge 3D is an advanced 3D reconstruction and analysis pipeline that transforms 2D videos into high-quality 4D Gaussian representations with transformer-based enhancement capabilities. The system provides comprehensive tools for depth estimation, 3D reconstruction, point cloud enhancement, and precise distance measurements.

### Key Features

- **Multi-Phase Pipeline**: Seamlessly integrates depth estimation, 4D Gaussian generation, and transformer enhancement
- **Advanced Algorithms**: Utilizes state-of-the-art depth estimation models and P4Transformer architecture
- **Interactive Visualization**: Comprehensive 3D visualization tools with temporal playback
- **Distance Measurement**: Precise 3D distance measurement tools with validation
- **Performance Optimization**: GPU acceleration and memory-efficient processing
- **Extensible Architecture**: Modular design for easy customization and extension

### System Requirements

**Minimum Requirements:**
- Python 3.8+
- 8GB RAM
- 2GB free disk space
- CPU with 4+ cores

**Recommended Requirements:**
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- 50GB+ free disk space for large projects
- CUDA 11.8+ for GPU acceleration

## Installation

### Option 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/gauge_3d.git
cd gauge_3d

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test.py
```

### Option 2: Development Install

```bash
# Clone with development dependencies
git clone https://github.com/yourusername/gauge_3d.git
cd gauge_3d

# Install in development mode
pip install -e .

# Install additional development tools
pip install -r requirements-dev.txt

# Run complete test suite
python test.py --verbose
```

### GPU Setup (Optional but Recommended)

```bash
# For NVIDIA GPUs with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Basic Usage - Single Video Processing

```bash
# Process a single video with default settings
python advanced_cli.py process input_video.mp4 --output results/

# Process with custom configuration
python advanced_cli.py process input_video.mp4 --config custom_config.yaml --output results/

# Process with GPU acceleration
python advanced_cli.py process input_video.mp4 --gpu --output results/
```

### Batch Processing Multiple Videos

```bash
# Process all MP4 files in a directory
python advanced_cli.py batch videos_folder/ --pattern "*.mp4" --output batch_results/

# Parallel processing with error handling
python advanced_cli.py batch videos_folder/ --parallel 4 --continue-on-error
```

### Interactive Distance Measurement

```bash
# Interactive 3D point selection for measurement
python advanced_cli.py measure results/enhanced/ --interactive --visualize

# Manual distance measurement
python advanced_cli.py measure results/enhanced/ --point1 100,200,50 --point2 150,250,75
```

### Visualization

```bash
# Interactive 3D visualization
python advanced_cli.py visualize results/ --type point_cloud --interactive

# Generate web dashboard
python advanced_cli.py visualize results/ --type dashboard --save results_dashboard.html
```

## Pipeline Overview

The Gauge 3D pipeline consists of four main phases:

### Phase 1: Depth Estimation
- **Input**: 2D video files (MP4, AVI, MOV, MKV)
- **Process**: Advanced monocular depth estimation using Depth Anything V2
- **Output**: Per-frame depth maps with confidence scores
- **Key Features**: 
  - Multiple model architectures (ViT-Small, ViT-Base, ViT-Large)
  - Automatic resolution optimization
  - Temporal consistency validation

### Phase 2: 4D Gaussian Generation  
- **Input**: Depth maps and original video frames
- **Process**: Conversion to 3D Gaussian representations with temporal coherence
- **Output**: 4D Gaussian splat data with positions, colors, scales, rotations, opacities
- **Key Features**:
  - Adaptive Gaussian initialization
  - Temporal consistency optimization
  - Quality-based filtering

### Phase 3: Transformer Enhancement
- **Input**: 4D Gaussian data
- **Process**: P4Transformer-based enhancement for improved quality and completeness
- **Output**: Enhanced point clouds and optimized Gaussian representations
- **Key Features**:
  - Point cloud denoising and completion
  - Feature enhancement and sharpening
  - Temporal consistency enforcement

### Phase 4: Integration and Analysis
- **Input**: Enhanced 3D representations
- **Process**: Distance measurement, visualization, and quality assessment
- **Output**: Measurements, visualizations, and analysis reports
- **Key Features**:
  - Precise 3D distance measurement
  - Interactive visualization tools
  - Comprehensive quality metrics

## Usage Examples

### Example 1: Basic Video Processing

```python
from unified_pipeline import UnifiedPipeline

# Initialize pipeline
pipeline = UnifiedPipeline()

# Process video
results = pipeline.run_full_pipeline(
    input_video="sample_video.mp4",
    output_dir="results/"
)

print(f"Processing completed in {results['processing_time']:.2f} seconds")
print(f"Generated {results['final_outputs']['summary_stats']['total_gaussians_generated']} Gaussians")
```

### Example 2: Custom Configuration

```python
# Create custom configuration
config = {
    "pipeline": {
        "enable_gpu": True,
        "batch_size": 8,
        "max_frames": 100
    },
    "depth_estimation": {
        "model": "depth_anything_v2_vitl14",
        "frame_skip": 3
    },
    "transformer": {
        "enhancement_enabled": True,
        "quality_threshold": 0.7
    }
}

# Save configuration
import yaml
with open("custom_config.yaml", "w") as f:
    yaml.dump(config, f)

# Use custom configuration
pipeline = UnifiedPipeline("custom_config.yaml")
results = pipeline.run_full_pipeline("video.mp4")
```

### Example 3: Interactive Distance Measurement

```python
from distance_measurement import DistanceMeasurementTool

# Initialize measurement tool
measurer = DistanceMeasurementTool()

# Load processed 3D data
measurer.load_point_cloud_data("results/enhanced/")

# Interactive measurement (opens 3D viewer)
measurement_result = measurer.interactive_measurement()

print(f"Distance: {measurement_result['distance']:.3f} units")
print(f"Accuracy: {measurement_result['accuracy_estimate']:.3f}")

# Export results
measurer.export_measurement(measurement_result, "measurement_report.json")
```

### Example 4: Batch Processing with Validation

```python
from pathlib import Path
from validation_suite import ValidationSuite

# Process multiple videos
input_videos = list(Path("input_videos/").glob("*.mp4"))
results = []

for video in input_videos:
    try:
        pipeline = UnifiedPipeline()
        result = pipeline.run_full_pipeline(video, f"results/{video.stem}/")
        results.append({"video": video, "success": True, "result": result})
    except Exception as e:
        results.append({"video": video, "success": False, "error": str(e)})

# Validate results
validator = ValidationSuite()
for result in results:
    if result["success"]:
        validation = validator.validate_pipeline_results(
            result["result"]["output_directory"]
        )
        print(f"{result['video'].name}: Quality Score = {validation['overall_quality']:.3f}")
```

### Example 5: Advanced Visualization

```python
from enhanced_visualization import Enhanced3DViewer, TemporalPlaybackViewer

# 3D visualization with measurements
viewer = Enhanced3DViewer("My 3D Scene")
viewer.load_point_cloud_sequence("results/enhanced/")

# Add measurement annotations
viewer.add_measurement_annotation(
    point1=(100, 200, 50),
    point2=(150, 250, 75),
    label="Distance A-B",
    color=(1.0, 0.0, 0.0)
)

# Show interactive viewer
viewer.show_interactive_3d()

# Temporal playback
temporal_viewer = TemporalPlaybackViewer("results/depth_maps/")
temporal_viewer.show_interactive_player()
```

## Configuration

### Master Configuration File

The pipeline uses a hierarchical YAML configuration system. Create `config/pipeline_config.yaml`:

```yaml
pipeline:
  input_video_path: "vinput/"
  output_base_path: "output/"
  enable_gpu: true
  batch_size: 4
  max_frames: null
  resume_from_checkpoint: true

depth_estimation:
  model: "depth_anything_v2_vitb14"
  frame_skip: 6
  output_format: "npy"
  confidence_threshold: 0.5

gaussian_4d:
  max_gaussians: 100000
  optimization_iterations: 1000
  temporal_consistency: true
  quality_threshold: 0.6

transformer:
  enhancement_enabled: true
  p4transformer_model: "placeholder"
  quality_threshold: 0.5
  denoising_strength: 0.8

validation:
  enable_quality_checks: true
  validate_outputs: true
  
performance:
  enable_profiling: false
  memory_limit_gb: 16
  parallel_processing: true
  gpu_memory_fraction: 0.8
```

### Phase-Specific Configurations

#### Depth Estimation Configuration
```yaml
# depth_estimation/config/dav_config.yaml
model:
  name: "depth_anything_v2_vitb14"
  input_size: [518, 518]
  checkpoint_path: null

processing:
  frame_skip: 6
  batch_size: 4
  enable_temporal_smoothing: true
  
output:
  format: "npy"
  save_visualizations: false
  compression: true
```

#### Transformer Configuration
```yaml
# transformer/config/transformer_config.yaml
extraction:
  min_points_per_frame: 1000
  max_points_per_frame: 50000
  quality_threshold: 0.5

enhancement:
  denoising:
    enabled: true
    method: "statistical"
    outlier_threshold: 2.0
  
  completion:
    enabled: true
    method: "surface_reconstruction"
    fill_holes: true
  
  temporal_consistency:
    enabled: true
    smoothing_window: 5
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA/GPU Issues

**Problem**: "CUDA out of memory" error
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
```bash
# Reduce batch size
python advanced_cli.py process video.mp4 --config config_small_batch.yaml

# Use CPU processing
python advanced_cli.py process video.mp4 --cpu

# Limit maximum frames
python advanced_cli.py process video.mp4 --max-frames 100
```

**Configuration fix**:
```yaml
pipeline:
  batch_size: 1
  enable_gpu: false
performance:
  gpu_memory_fraction: 0.5
```

#### 2. Import/Dependency Issues

**Problem**: "ModuleNotFoundError: No module named 'torch'"

**Solution**:
```bash
# Reinstall PyTorch
pip install torch torchvision torchaudio

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.__version__)"
```

#### 3. Memory Issues

**Problem**: "MemoryError" or system freezing

**Solutions**:
```bash
# Process smaller segments
python advanced_cli.py process video.mp4 --max-frames 50

# Enable memory optimization
python advanced_cli.py process video.mp4 --config memory_optimized.yaml
```

**Memory-optimized configuration**:
```yaml
pipeline:
  batch_size: 1
performance:
  memory_limit_gb: 8
  parallel_processing: false
gaussian_4d:
  max_gaussians: 50000
```

#### 4. Video Format Issues

**Problem**: "Unsupported video format" or codec errors

**Solutions**:
```bash
# Convert video using ffmpeg
ffmpeg -i input_video.mov -c:v libx264 -c:a aac output_video.mp4

# Check supported formats
python advanced_cli.py process --help
```

#### 5. Quality Issues

**Problem**: Poor reconstruction quality

**Solutions**:
1. **Adjust quality thresholds**:
```yaml
depth_estimation:
  confidence_threshold: 0.7
gaussian_4d:
  quality_threshold: 0.8
transformer:
  quality_threshold: 0.7
```

2. **Use higher-quality models**:
```yaml
depth_estimation:
  model: "depth_anything_v2_vitl14"  # Larger model
```

3. **Increase processing density**:
```yaml
depth_estimation:
  frame_skip: 3  # Process more frames
gaussian_4d:
  max_gaussians: 200000  # More detail
```

### Performance Optimization

#### 1. GPU Optimization

```yaml
performance:
  enable_gpu: true
  gpu_memory_fraction: 0.9
  batch_size: 8
  parallel_processing: true
```

```python
# Monitor GPU usage
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU Utilization: {torch.cuda.utilization()}%")
```

#### 2. Memory Optimization

```yaml
pipeline:
  # Process in smaller chunks
  max_frames: 100
  
performance:
  # Limit memory usage
  memory_limit_gb: 12
  
  # Enable memory mapping for large files
  use_memory_mapping: true
```

#### 3. CPU Optimization

```yaml
performance:
  parallel_processing: true
  num_workers: 4  # Adjust based on CPU cores
  
pipeline:
  batch_size: 2  # Lower for CPU processing
```

#### 4. Storage Optimization

```yaml
output:
  # Compress intermediate files
  compression: true
  
  # Save only essential outputs
  save_intermediate: false
  
pipeline:
  # Use temporary storage for processing
  temp_path: "/tmp/gauge3d/"
```

### Debugging and Logging

#### Enable Verbose Logging

```bash
# CLI verbose mode
python advanced_cli.py process video.mp4 --verbose

# Python API
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

#### Performance Profiling

```bash
# Generate performance report
python advanced_cli.py profile results/ --benchmark --memory --gpu --report performance_report.html
```

#### Validation and Testing

```bash
# Validate pipeline results
python advanced_cli.py validate results/ --metrics depth_accuracy reconstruction_quality

# Run comprehensive test suite
python test.py --verbose

# Test specific components
python test.py --module transformer
```

## API Reference

### Core Classes

#### UnifiedPipeline
```python
class UnifiedPipeline:
    def __init__(self, config_path: Optional[str] = None)
    def run_full_pipeline(self, input_video: str, output_dir: str) -> Dict
    def run_depth_estimation(self, input_video: Path) -> Dict
    def run_gaussian_generation(self, depth_results: Dict) -> Dict
    def run_transformer_enhancement(self, gaussian_results: Dict) -> Dict
```

#### DistanceMeasurementTool
```python
class DistanceMeasurementTool:
    def __init__(self, precision: float = 0.001)
    def interactive_measurement(self, data_dir: str) -> Dict
    def measure_distance(self, data_dir: str, point1: Tuple, point2: Tuple) -> Dict
    def export_measurement(self, result: Dict, output_file: str)
```

#### Enhanced3DViewer
```python
class Enhanced3DViewer:
    def __init__(self, title: str = "Gauge 3D Viewer")
    def load_point_cloud_sequence(self, data_dir: str) -> bool
    def add_measurement_annotation(self, point1: Tuple, point2: Tuple, label: str)
    def show_interactive_3d(self, enable_animation: bool = True)
```

### Configuration Management

```python
from unified_pipeline import ConfigurationManager

# Load configuration
config_manager = ConfigurationManager("config/pipeline_config.yaml")

# Get phase-specific config
depth_config = config_manager.get_phase_config("depth_estimation")

# Update configuration
config_manager.config["pipeline"]["batch_size"] = 8
config_manager.save_configuration()
```

### Error Handling

```python
from unified_pipeline import PipelineError

try:
    pipeline = UnifiedPipeline()
    results = pipeline.run_full_pipeline("video.mp4")
except PipelineError as e:
    print(f"Pipeline failed: {e}")
    # Handle error or retry with different settings
```

## Contributing

We welcome contributions to Gauge 3D! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/gauge_3d.git
cd gauge_3d

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests before making changes
python test.py --verbose

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Full test suite
python test.py

# Specific test modules
python test.py --module depth
python test.py --module transformer

# Integration tests
python test_transformer_pipeline.py
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass
5. Submit pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://gauge3d.readthedocs.io](https://gauge3d.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/gauge_3d/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gauge_3d/discussions)
- **Email**: support@gauge3d.com

## Citation

If you use Gauge 3D in your research, please cite:

```bibtex
@software{gauge3d2025,
  title={Gauge 3D: Advanced 3D Reconstruction and Analysis Pipeline},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/gauge_3d}
}
```
