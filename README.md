# Gauge 3D - Video to 3D Reconstruction Pipeline

A comprehensive Python pipeline for converting videos into 3D reconstructions through depth estimation and 4D Gaussian splatting. Features interactive visualization tools and end-to-end processing from video frames to volumetric representations.

## Features

### Core Pipeline
- **Video Processing**: Extract frames with OpenCV and FFmpeg support
- **Depth Estimation**: Generate depth maps using state-of-the-art models
- **4D Gaussian Splatting**: Convert depth data to volumetric 3D representations
- **Interactive Visualization**: Real-time depth map viewer with frame navigation

### Processing Capabilities
- **Hybrid Processing**: Automatic selection between OpenCV and FFmpeg
- **Smart Sampling**: Uniform and adaptive frame sampling
- **High Performance**: Optimized for large video datasets
- **Quality Control**: Configurable output quality and resolution

### Visualization Tools
- **Depth Map Viewer**: Interactive frame-by-frame navigation
- **Real-time Display**: Live depth visualization with statistics
- **Export Options**: Save depth maps and visualizations

## Installation

This project uses `uv` for dependency management and requires several ML/CV libraries.

### Prerequisites

```bash
# Install uv (if not already installed)
pip install uv

# Install FFmpeg (recommended for video processing)
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg

# Windows:
# Download from https://ffmpeg.org/download.html
```

### Project Setup

```bash
# Clone the repository
git clone <repository-url>
cd gauge_3d

# Install Python dependencies
uv sync

# Set up environment variables (create .env file)
echo 'HUGGINGFACE_HUB_TOKEN="your_token_here"' > .env
```

## Quick Start

### 1. Extract Video Frames
```bash
# Extract frames from video
python run.py extract video.mp4

# Extract key frames for analysis
python run.py extract video.mp4 --key-frames 20
```

### 2. Generate Depth Maps
```bash
# Process extracted frames to generate depth maps
python depth_estimation/main_pipeline.py --input vinput/video.mp4 --output output/
```

### 3. View Depth Maps Interactively
```bash
# Launch interactive depth viewer
python viewer/depth_viewer.py --dataset video_name

# View specific frame
python viewer/depth_viewer.py --dataset video_name --frame 100
```

### 4. Run Tests
```bash
# Run all tests
python test.py

# Test specific components
python test.py --section environment
python test.py --section depth
python test.py --section visualization
```

## Advanced Usage

### Video Frame Extraction
```bash
# Extract every 30th frame (default)
python main.py video.mp4

# Extract every 10th frame
python main.py video.mp4 --frame-interval 10

# Extract specific range of frames
python main.py video.mp4 --start-frame 100 --end-frame 500

# Extract key frames uniformly distributed
python main.py video.mp4 --key-frames 50

# Use specific processing engine
python main.py video.mp4 --engine ffmpeg --frame-interval 30
```

### Depth Estimation Pipeline
```bash
# Process video with depth estimation
python depth_estimation/main_pipeline.py --input vinput/video.mp4 --output output/

# Custom depth model configuration
python depth_estimation/main_pipeline.py --input video.mp4 --config depth_estimation/config/dav_config.yaml
```

### Interactive Depth Visualization
```bash
# View depth maps with navigation controls
python viewer/depth_viewer.py --dataset 1080_60_fps

# Start from specific frame
python viewer/depth_viewer.py --dataset 1080_60_fps --frame 150
```

### Testing and Validation
```bash
# Run comprehensive test suite
python test.py

# Test specific components
python test.py --section environment     # Check dependencies
python test.py --section depth         # Test depth estimation
python test.py --section gaussian      # Test 4D Gaussian components
python test.py --section visualization # Test viewers
python test.py --section integration   # Test component integration

# Verbose testing output
python test.py --verbose
```

## Project Structure

```
gauge_3d/
├── README.md              # Project documentation
├── pyproject.toml         # Dependencies and project config
├── test.py               # Consolidated test suite
├── main.py               # Video frame extraction
├── run.py                # Simplified interface
├── utils.py              # Analysis utilities
│
├── src/                  # Core video processing
│   └── video_processor.py
│
├── depth_estimation/     # Depth map generation
│   ├── depth_pipeline.py
│   ├── main_pipeline.py
│   └── config/
│       └── dav_config.yaml
│
├── viewer/               # Visualization tools
│   ├── __init__.py
│   └── depth_viewer.py   # Interactive depth map viewer
│
├── 4d_gaussian/          # 4D Gaussian splatting
│   ├── config/
│   ├── data_preparation/
│   ├── gaussian_generation/
│   └── run_4d_gaussian.py
│
├── vinput/               # Input videos
│   ├── 1080_60_fps.mp4
│   └── 4k_30_fps.mp4
│
└── output/               # Generated outputs
    ├── frames/           # Extracted video frames
    ├── depth_maps/       # Generated depth maps
    ├── depth_previews/   # Depth visualization exports
    └── gaussian_reconstruction/  # 4D Gaussian outputs
```

## Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework for depth estimation
- **OpenCV**: Video processing and computer vision
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Pillow**: Image processing
- **Tkinter**: GUI framework for interactive viewers

### Machine Learning
- **Transformers**: Hugging Face transformer models
- **Diffusers**: Diffusion models for depth estimation
- **Kornia**: Computer vision operations
- **Trimesh**: 3D mesh processing

### Visualization
- **Seaborn**: Statistical data visualization
- **Pandas**: Data analysis and manipulation
- **Plotly**: Interactive plotting (optional)

### Optional Performance
- **FFmpeg**: High-performance video processing
- **Open3D**: 3D data processing (optional)

## Interactive Depth Viewer

The depth viewer provides real-time navigation through depth map sequences:

### Navigation Controls
- **Arrow Keys**: Previous/Next frame navigation
- **Mouse**: Click navigation buttons
- **Slider**: Scrub through entire sequence
- **Jump**: Enter specific frame numbers
- **Keyboard Shortcuts**:
  - `←` / `→`: Navigate frames
  - `Enter`: Jump to entered frame number

### Features
- Real-time depth statistics (min/max/mean)
- Automatic image scaling for display
- Frame information overlay
- Smooth navigation with large datasets

## Testing

The project includes a comprehensive test suite that validates:

### Environment Tests
- Python version compatibility
- Core dependency availability
- CUDA/GPU detection
- System requirements

### Component Tests
- Depth estimation pipeline
- 4D Gaussian data structures
- Visualization tools
- File I/O operations

### Integration Tests
- End-to-end pipeline functionality
- Output directory management
- Cross-component compatibility

Run tests regularly to ensure system health:
```bash
python test.py  # Run all tests
```

## Workflow Examples

### Complete 3D Reconstruction Pipeline
```bash
# 1. Extract frames from video
python run.py extract input_video.mp4 --key-frames 50

# 2. Generate depth maps
python depth_estimation/main_pipeline.py --input vinput/input_video.mp4

# 3. Visualize results interactively
python viewer/depth_viewer.py --dataset input_video

# 4. Run quality checks
python test.py --section depth
```

### Quick Analysis Workflow
```bash
# Extract representative frames for quick analysis
python main.py video.mp4 --key-frames 20

# View depth estimation results
python viewer/depth_viewer.py

# Check system health
python test.py --section environment
```

### High-Quality Production Workflow
```bash
# Extract high-quality frames
python main.py video.mp4 --frame-interval 5 --quality 100 --resize 1920 1080

# Process with depth estimation
python depth_estimation/main_pipeline.py --input video.mp4 --output production_output/

# Validate results
python test.py --section integration
```

## Supported Formats

### Video Input
- MP4 (.mp4) - Recommended
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)
- WMV (.wmv)
- FLV (.flv)

### Output Formats
- **Frames**: JPEG, PNG
- **Depth Maps**: NumPy arrays (.npy)
- **Visualizations**: PNG, JPEG
- **3D Data**: PLY, OBJ (future)

## Performance Optimization

### Hardware Recommendations
- **GPU**: CUDA-compatible for depth estimation acceleration
- **RAM**: 16GB+ for large video processing
- **Storage**: SSD recommended for frame I/O operations

### Processing Tips
- Use key frame extraction for faster initial analysis
- Enable GPU acceleration when available
- Process videos in smaller segments for memory efficiency
- Use FFmpeg for bulk frame extraction (faster than OpenCV)

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
# Check all dependencies
python test.py --section environment

# Install missing packages
uv sync
```

#### CUDA/GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run CPU-only mode if needed
export CUDA_VISIBLE_DEVICES=""
```

#### Memory Issues
```bash
# Process smaller frame batches
python main.py video.mp4 --frame-interval 30  # Fewer frames

# Monitor memory usage
python test.py --section integration --verbose
```

#### Visualization Issues
```bash
# Test GUI dependencies
python test.py --section visualization

# Use alternative display backend
export MPLBACKEND=TkAgg
```

## Development

### Code Organization
- **Modular Design**: Each component is independently testable
- **Configuration-Driven**: YAML configs for pipeline parameters
- **Error Handling**: Comprehensive error reporting and recovery
- **Logging**: Detailed logging for debugging and monitoring

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python test.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Testing Guidelines
- Run full test suite before submitting PRs
- Add tests for new functionality
- Ensure GPU and CPU compatibility
- Test with different video formats and sizes

## Future Roadmap

### Phase 2: Enhanced 3D Reconstruction
- ✅ Depth estimation pipeline
- ✅ Interactive visualization
- 🔄 4D Gaussian splatting optimization
- 📅 Real-time processing

### Phase 3: Advanced Features
- 📅 Multi-view reconstruction
- 📅 Temporal consistency optimization
- 📅 Object tracking and measurement
- 📅 Export to standard 3D formats

### Phase 4: Production Features
- 📅 Web interface
- 📅 Cloud processing integration
- 📅 Real-time streaming support
- 📅 Mobile app integration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Depth estimation models from Hugging Face
- 4D Gaussian Splatting research community
- OpenCV and FFmpeg communities
- PyTorch ecosystem contributors
