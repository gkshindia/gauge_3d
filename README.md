# Gauge 3D - Video to 3D Reconstruction and Distance Measurement

A Python application that processes video files to extract frames, perform 3D reconstruction, generate point clouds, and measure distances between objects.

## Features

- **Video Frame Extraction**: Extract frames from various video formats
- **Smart Sampling**: Support for uniform and adaptive frame sampling
- **Key Frame Detection**: Extract representative frames for efficient processing
- **Flexible Output**: Configurable frame intervals, quality, and resizing
- **Multiple Formats**: Support for MP4, AVI, MOV, MKV, and more

## Installation

This project uses `uv` for dependency management. Make sure you have `uv` installed:

```bash
# Install uv (if not already installed)
pip install uv

# Clone the repository
git clone <repository-url>
cd gauge_3d

# Install dependencies
uv sync
```

## Usage

### Quick Start (Simplified Interface)

```bash
# Extract frames from video (every 30th frame by default)
python run.py extract video.mp4

# Extract 20 key frames
python run.py extract video.mp4 --key-frames 20

# Get video information
python run.py info video.mp4

# List all videos and extractions
python run.py list

# Analyze workspace
python run.py analyze
```

### Advanced Usage (Full Interface)

```bash
# Extract every 30th frame (default)
python main.py video.mp4

# Extract every 10th frame
python main.py video.mp4 --frame-interval 10

# Extract specific range of frames
python main.py video.mp4 --start-frame 100 --end-frame 500
```

### Key Frame Extraction

```bash
# Extract 50 key frames uniformly distributed
python main.py video.mp4 --key-frames 50

# Extract key frames with adaptive sampling
python main.py video.mp4 --key-frames 30 --method adaptive
```

### Advanced Options

```bash
# Resize frames and save with custom quality
python main.py video.mp4 --resize 1920 1080 --quality 85

# Custom output directory
python main.py video.mp4 --output-dir ./my_frames

# Clean output directory before extraction
python main.py video.mp4 --clean

# Show video information only
python main.py video.mp4 --info
```

### Command Line Options

- `video_path`: Path to the input video file (required)
- `--output-dir, -o`: Output directory for extracted frames (default: output/frames)
- `--frame-interval, -i`: Extract every nth frame (default: 30)
- `--start-frame, -s`: Frame number to start extraction (default: 0)
- `--end-frame, -e`: Frame number to end extraction (default: end of video)
- `--resize, -r`: Resize frames to specified WIDTH HEIGHT
- `--quality, -q`: JPEG quality for saved frames (0-100, default: 95)
- `--key-frames, -k`: Extract NUM key frames instead of regular extraction
- `--method, -m`: Key frame sampling method (uniform, adaptive)
- `--clean`: Clean output directory before extraction
- `--info`: Show video information only, don't extract frames
- `--verbose, -v`: Enable verbose output

## Project Structure

```
gauge_3d/
├── main.py                 # Full-featured main entry point
├── run.py                  # Simplified quick commands interface
├── utils.py                # Utility functions and analysis tools
├── src/
│   ├── __init__.py
│   └── video_processor.py  # Core frame extraction functionality
├── vinput/                 # Input videos directory
│   ├── 1080_60_fps.mp4
│   └── 4k_30_fps.mp4
├── output/                 # Generated output
│   └── frames/            # Extracted frames organized by video
│       ├── video_name_1/
│       └── video_name_2/
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock                # Dependency lock file
└── README.md              # This file
```

## Available Scripts

- **`run.py`**: Simplified interface for common operations
- **`main.py`**: Full-featured frame extraction with all options
- **`utils.py`**: Utilities for analysis and workspace management

## Dependencies

- **opencv-python**: Video processing and frame extraction
- **numpy**: Numerical operations
- **Pillow**: Image processing
- **tqdm**: Progress bars

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)
- WebM (.webm)
- M4V (.m4v)

## Examples

### Extract frames for 3D reconstruction
```bash
# For stereo vision or SfM, extract every 5th frame
python main.py video.mp4 --frame-interval 5 --resize 1920 1080
```

### Quick preview with key frames
```bash
# Extract 20 representative frames for quick analysis
python main.py video.mp4 --key-frames 20
```

### High-quality extraction for detailed analysis
```bash
# Extract frames with maximum quality
python main.py video.mp4 --frame-interval 1 --quality 100
```

## Future Enhancements

This is the first phase of the project. Future modules will include:

- 3D reconstruction algorithms (Structure from Motion, Stereo Vision)
- Point cloud generation and processing
- Distance measurement tools
- Object detection and tracking
- Real-time processing capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
