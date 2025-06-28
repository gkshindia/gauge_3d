# Depth Estimation Pipeline - Phase 1

This module implements a complete video-to-depth estimation pipeline using DepthAnyVideo (DA-V) for Phase 1 of the 3D reconstruction project.

## Overview

The Phase 1 pipeline consists of four main steps:

1. **Step 1.1: Environment Setup** - Automated setup and verification of DA-V environment
2. **Step 1.2: Video Preprocessing** - Frame extraction with quality assessment and stabilization
3. **Step 1.3: Depth Estimation** - Frame-by-frame depth estimation using DA-V
4. **Step 1.4: Output Management** - Depth map validation, organization, and archiving

## Project Structure

```
depth_estimation/
├── main_pipeline.py          # Main entry point for the complete pipeline
├── setup_environment.py      # Environment setup and verification
├── video_depth_processor.py  # Video preprocessing with quality assessment
├── depth_pipeline.py         # DA-V depth estimation implementation
├── output_manager.py         # Output organization and validation
├── models/                   # Directory for model weights
├── utils/
│   └── depth_utils.py       # Utility functions for depth operations
└── README.md                # This file
```

## Features

### Step 1.1: Environment Setup
- Automated DA-V model download and setup
- GPU/CPU configuration and optimization
- Dependency verification and installation
- Environment validation

### Step 1.2: Video Preprocessing
- High-quality frame extraction with customizable intervals
- Frame quality assessment (blur, brightness, motion)
- Optional video stabilization (optical flow or feature-based)
- Batch processing support
- Quality filtering and reporting

### Step 1.3: Depth Estimation
- Integration with DepthAnyVideo (DA-V) models
- Support for both base and large model variants
- Batch processing for efficiency
- Temporal consistency filtering
- Multiple output formats (NPY, PNG, EXR)
- Comprehensive depth map validation

### Step 1.4: Output Management
- Organized directory structure with timestamps
- Quality metrics calculation and reporting
- Depth map visualization and previews
- Comprehensive metadata generation
- Archiving and compression options
- Batch processing summaries

## Quick Start

### 1. Environment Setup

First, set up the environment:

```bash
cd depth_estimation
python main_pipeline.py --setup-only
```

### 2. Process a Single Video

```bash
python main_pipeline.py path/to/video.mp4 --output-dir ./results
```

### 3. Process Multiple Videos

```bash
python main_pipeline.py video1.mp4 video2.mp4 video3.mp4 --output-dir ./batch_results
```

### 4. Advanced Processing

```bash
python main_pipeline.py video.mp4 \
    --frame-interval 5 \
    --enable-stabilization \
    --depth-model depth-anything-v2-large \
    --batch-size 8 \
    --depth-format png \
    --output-dir ./advanced_results
```

## Configuration Options

### Video Processing Configuration

```python
ProcessingConfig(
    frame_interval=1,                    # Extract every nth frame
    max_frames=1000,                     # Maximum frames to process
    target_fps=None,                     # Target FPS for extraction
    resize_target=None,                  # Resize frames to (width, height)
    blur_threshold=100.0,                # Blur detection threshold
    brightness_range=(20.0, 235.0),     # Acceptable brightness range
    motion_threshold=50.0,               # Motion detection threshold
    enable_stabilization=False,          # Enable video stabilization
    stabilization_method="optical_flow"  # Stabilization method
)
```

### Depth Estimation Configuration

```python
DepthConfig(
    model_name="depth-anything-v2-base",  # Model variant
    device="auto",                        # Device selection
    batch_size=4,                         # Batch size for processing
    max_resolution=(1024, 1024),          # Maximum input resolution
    output_format="npy",                  # Output format
    normalize_depth=True,                 # Normalize depth values
    apply_temporal_consistency=True,      # Enable temporal filtering
    temporal_window=5                     # Temporal filter window size
)
```

### Output Management Configuration

```python
OutputConfig(
    base_output_dir="output",           # Base output directory
    organize_by_date=True,              # Create timestamped directories
    create_video_subdirs=True,          # Create per-video subdirectories
    depth_format="npy",                 # Depth map format
    preview_format="jpg",               # Preview image format
    enable_quality_validation=True,     # Enable quality assessment
    enable_archiving=True,              # Create result archives
    compress_archives=True              # Compress archives
)
```

## Command Line Interface

```bash
usage: main_pipeline.py [-h] [--output-dir OUTPUT_DIR] [--frame-interval FRAME_INTERVAL]
                       [--max-frames MAX_FRAMES] [--target-fps TARGET_FPS]
                       [--resize WIDTH HEIGHT] [--enable-stabilization]
                       [--blur-threshold BLUR_THRESHOLD]
                       [--brightness-range BRIGHTNESS_RANGE BRIGHTNESS_RANGE]
                       [--depth-model {depth-anything-v2-base,depth-anything-v2-large}]
                       [--batch-size BATCH_SIZE] [--max-resolution MAX_RESOLUTION MAX_RESOLUTION]
                       [--disable-temporal-consistency] [--depth-format {npy,exr,png}]
                       [--no-archive] [--keep-intermediates] [--setup-only] [--verify-setup]
                       [videos ...]

Video Depth Estimation Pipeline - Phase 1

positional arguments:
  videos                Input video file(s)

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for all results
  --frame-interval FRAME_INTERVAL
                        Extract every nth frame
  --max-frames MAX_FRAMES
                        Maximum number of frames to process
  --target-fps TARGET_FPS
                        Target FPS for frame extraction
  --resize WIDTH HEIGHT
                        Resize frames to specified dimensions
  --enable-stabilization
                        Enable video stabilization
  --blur-threshold BLUR_THRESHOLD
                        Blur detection threshold
  --brightness-range BRIGHTNESS_RANGE BRIGHTNESS_RANGE
                        Acceptable brightness range
  --depth-model {depth-anything-v2-base,depth-anything-v2-large}
                        Depth estimation model
  --batch-size BATCH_SIZE
                        Batch size for depth estimation
  --max-resolution MAX_RESOLUTION MAX_RESOLUTION
                        Maximum resolution for depth estimation
  --disable-temporal-consistency
                        Disable temporal consistency filtering
  --depth-format {npy,exr,png}
                        Output format for depth maps
  --no-archive          Disable archiving of results
  --keep-intermediates  Keep intermediate processing files
  --setup-only          Only setup environment, don't process videos
  --verify-setup        Verify environment setup before processing
```

## Output Structure

The pipeline creates a comprehensive output structure:

```
output/
├── session_YYYYMMDD_HHMMSS/
│   ├── frames/
│   │   └── video_name/
│   │       ├── frame_000001.jpg
│   │       ├── frame_000002.jpg
│   │       └── ...
│   ├── depth_maps/
│   │   └── video_name/
│   │       ├── depth_000001.npy
│   │       ├── depth_000002.npy
│   │       └── ...
│   ├── previews/
│   │   └── video_name/
│   │       ├── frame_000001_preview.jpg
│   │       ├── frame_000002_preview.jpg
│   │       └── ...
│   ├── quality_reports/
│   │   └── video_name_quality_report.json
│   ├── visualizations/
│   │   └── video_name_quality_report.png
│   ├── metadata/
│   │   └── video_name/
│   │       ├── frame_000001_metadata.json
│   │       ├── frame_000002_metadata.json
│   │       └── video_name_summary.json
│   └── archives/
│       └── video_name_results.tar.gz
```

## API Usage

### Programmatic Usage

```python
from depth_estimation.main_pipeline import DepthEstimationMaster
from depth_estimation.video_depth_processor import create_default_config
from depth_estimation.depth_pipeline import create_default_depth_config
from depth_estimation.output_manager import create_default_output_config

# Create configurations
processing_config = create_default_config()
processing_config.frame_interval = 5
processing_config.enable_stabilization = True

depth_config = create_default_depth_config()
depth_config.model_name = "depth-anything-v2-large"
depth_config.batch_size = 8

output_config = create_default_output_config()
output_config.base_output_dir = "./my_results"

# Initialize pipeline
pipeline = DepthEstimationMaster(
    processing_config=processing_config,
    depth_config=depth_config,
    output_config=output_config,
    output_dir="./my_results"
)

# Process video
results = pipeline.process_video("path/to/video.mp4")

if results['overall_success']:
    print(f"Successfully processed {results['frames_processed_for_depth']} frames")
    print(f"Output directory: {results['pipeline_stages']['output_management']['output_dirs']}")
else:
    print(f"Processing failed: {results['error_message']}")
```

### Individual Component Usage

```python
# Use video processor only
from depth_estimation.video_depth_processor import VideoDepthProcessor, create_default_config

config = create_default_config()
processor = VideoDepthProcessor(config, "./output")
results = processor.preprocess_video("video.mp4")

# Use depth pipeline only
from depth_estimation.depth_pipeline import DepthEstimationPipeline, create_default_depth_config

config = create_default_depth_config()
pipeline = DepthEstimationPipeline(config, "./output")
frame_paths = ["frame1.jpg", "frame2.jpg", ...]
results = pipeline.process_video_sequence(frame_paths, "video_name")

# Use output manager only
from depth_estimation.output_manager import DepthOutputManager, create_default_output_config

config = create_default_output_config()
manager = DepthOutputManager(config)
summary = manager.process_video_output("video_name", frame_data, original_frames)
```

## Quality Metrics

The pipeline calculates comprehensive quality metrics:

### Frame Quality Assessment
- **Blur Score**: Laplacian variance for sharpness assessment
- **Brightness**: Average pixel intensity
- **Motion Score**: Optical flow magnitude between frames
- **Overall Acceptance**: Combined quality score

### Depth Map Quality Metrics
- **Smoothness Score**: Inverse of gradient variance
- **Consistency Score**: Local smoothness assessment
- **Coverage Ratio**: Ratio of valid depth pixels
- **Temporal Consistency**: Frame-to-frame consistency
- **Overall Quality**: Weighted combination of all metrics

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in depth configuration
   - Reduce `max_resolution` for depth estimation
   - Enable frame limiting with `max_frames`

2. **Model Download Fails**
   - Check internet connection
   - Run setup with `--setup-only` flag
   - Verify Hugging Face access

3. **Poor Quality Results**
   - Adjust quality thresholds in processing configuration
   - Enable video stabilization
   - Check input video quality

4. **Slow Processing**
   - Increase `frame_interval` to process fewer frames
   - Use smaller depth model (`depth-anything-v2-base`)
   - Reduce input resolution with `--resize`

### Performance Optimization

- **GPU Memory**: Monitor usage with `nvidia-smi`
- **Batch Size**: Start with 4, increase if memory allows
- **Frame Interval**: Use 5-10 for fast processing
- **Resolution**: Limit to 1024x1024 for speed
- **Temporal Filtering**: Disable for faster processing

## Dependencies

Core dependencies are managed in the parent `pyproject.toml`:

- torch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- diffusers >= 0.20.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scikit-image >= 0.21.0
- Pillow >= 10.0.0
- tqdm >= 4.65.0

## Integration with Phase 2

The depth estimation outputs are designed for seamless integration with Phase 2 (4D Gaussian Splatting):

- **Depth Maps**: Saved in multiple formats for compatibility
- **Frame Synchronization**: Maintained through consistent naming
- **Quality Metrics**: Provide guidance for 3D reconstruction
- **Temporal Consistency**: Prepares data for dynamic scene modeling

## License

This project is part of the gauge-3d pipeline. See the main project LICENSE for details.

## Contributing

1. Follow the established code structure
2. Add comprehensive logging
3. Include quality metrics for new features
4. Update this README for significant changes
5. Test with various video formats and resolutions

## Future Enhancements

- [ ] Support for stereo video processing
- [ ] Advanced temporal consistency algorithms
- [ ] Real-time processing capabilities
- [ ] Integration with other depth estimation models
- [ ] Web-based interface for monitoring
- [ ] Distributed processing support
