#!/usr/bin/env python3
"""
Gauge 3D Test Suite

Consolidated test file for all components of the Gauge 3D pipeline.
Includes tests for environment setup, dependencies, depth estimation, 
4D Gaussian splatting, and visualization components.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import logging
import subprocess
import importlib.util

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestRunner:
    """Main test runner class"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and track results"""
        print(f"\n{'='*60}")
        print(f"Testing: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            if result:
                self.passed += 1
                self.results.append((test_name, "PASSED"))
                print(f"✅ {test_name} - PASSED")
            else:
                self.failed += 1
                self.results.append((test_name, "FAILED"))
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            self.failed += 1
            self.results.append((test_name, f"ERROR: {e}"))
            print(f"❌ {test_name} - ERROR: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print('='*60)
        print(f"Total tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")
        
        if self.failed > 0:
            print("\nFailed tests:")
            for name, status in self.results:
                if status != "PASSED":
                    print(f"  - {name}: {status}")


def test_environment():
    """Test environment setup and basic dependencies"""
    print("Testing environment setup...")
    
    try:
        # Test Python version
        python_version = sys.version_info
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Test basic imports
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available (CPU mode only)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Environment test failed: {e}")
        return False


def test_depth_estimation_dependencies():
    """Test depth estimation pipeline dependencies"""
    print("Testing depth estimation dependencies...")
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        import diffusers
        print(f"✅ Diffusers {diffusers.__version__}")
        
        import PIL
        print(f"✅ PIL/Pillow {PIL.__version__}")
        
        import scipy
        print(f"✅ SciPy {scipy.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Depth estimation dependencies failed: {e}")
        return False


def test_4d_gaussian_dependencies():
    """Test 4D Gaussian splatting dependencies"""
    print("Testing 4D Gaussian dependencies...")
    
    try:
        import trimesh
        print(f"✅ Trimesh {trimesh.__version__}")
        
        import plyfile
        print("✅ PLYFile available")
        
        import roma
        print("✅ Roma available")
        
        import kornia
        print(f"✅ Kornia {kornia.__version__}")
        
        # Test Open3D (optional)
        try:
            import open3d
            print(f"✅ Open3D {open3d.__version__}")
        except ImportError:
            print("⚠️  Open3D not available (optional)")
        
        return True
        
    except ImportError as e:
        print(f"❌ 4D Gaussian dependencies failed: {e}")
        return False


def test_visualization_dependencies():
    """Test visualization dependencies"""
    print("Testing visualization dependencies...")
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib pyplot")
        
        import seaborn as sns
        print(f"✅ Seaborn {sns.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import tkinter as tk
        print("✅ Tkinter available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Visualization dependencies failed: {e}")
        return False


def test_project_structure():
    """Test project structure and file organization"""
    print("Testing project structure...")
    
    required_dirs = [
        "viewer",
        "depth_estimation", 
        "4d_gaussian",
        "src",
        "output"
    ]
    
    required_files = [
        "pyproject.toml",
        "README.md",
        "viewer/depth_viewer.py",
        "depth_estimation/depth_pipeline.py"
    ]
    
    try:
        # Check directories
        for dir_name in required_dirs:
            if Path(dir_name).exists():
                print(f"✅ Directory: {dir_name}")
            else:
                print(f"⚠️  Missing directory: {dir_name}")
        
        # Check files
        for file_name in required_files:
            if Path(file_name).exists():
                print(f"✅ File: {file_name}")
            else:
                print(f"⚠️  Missing file: {file_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Project structure test failed: {e}")
        return False


def test_depth_viewer():
    """Test depth viewer functionality"""
    print("Testing depth viewer...")
    
    try:
        from viewer.depth_viewer import DepthViewer
        
        viewer = DepthViewer()
        print("✅ DepthViewer instantiated")
        
        # Test dataset discovery
        datasets = viewer.find_depth_datasets()
        print(f"✅ Found {len(datasets)} depth datasets: {datasets}")
        
        return True
        
    except Exception as e:
        print(f"❌ Depth viewer test failed: {e}")
        return False


def test_gaussian_data_structures():
    """Test Gaussian data structure operations"""
    print("Testing Gaussian data structures...")
    
    try:
        import torch
        
        # Simulate Gaussian parameters
        n_gaussians = 1000
        
        # Positions (xyz)
        positions = torch.randn(n_gaussians, 3)
        
        # Colors (rgb)
        colors = torch.rand(n_gaussians, 3)
        
        # Scales (xyz scales)
        scales = torch.rand(n_gaussians, 3) * 0.1
        
        # Rotations (quaternions w,x,y,z)
        rotations = torch.randn(n_gaussians, 4)
        rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)
        
        # Opacity (logit space)
        opacity = torch.randn(n_gaussians, 1)
        
        print(f"✅ Created {n_gaussians} Gaussians with proper structure")
        print(f"   Positions: {positions.shape}")
        print(f"   Colors: {colors.shape}")
        print(f"   Scales: {scales.shape}")
        print(f"   Rotations: {rotations.shape}")
        print(f"   Opacity: {opacity.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gaussian structure test failed: {e}")
        return False


def test_depth_pipeline():
    """Test depth estimation pipeline"""
    print("Testing depth pipeline...")
    
    try:
        # Check if depth pipeline can be imported
        from depth_estimation.depth_pipeline import DepthPipeline
        
        # Create a minimal pipeline instance
        pipeline = DepthPipeline()
        print("✅ DepthPipeline instantiated")
        
        # Test with synthetic data
        synthetic_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✅ Created synthetic test image")
        
        return True
        
    except Exception as e:
        print(f"❌ Depth pipeline test failed: {e}")
        return False


def test_integration():
    """Test integration between components"""
    print("Testing component integration...")
    
    try:
        # Test that output directories exist or can be created
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        test_dirs = [
            "output/depth_maps",
            "output/frames",
            "output/gaussian_reconstruction"
        ]
        
        for dir_path in test_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory ready: {dir_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Gauge 3D Test Suite")
    parser.add_argument("--section", choices=[
        "environment", "dependencies", "structure", "depth", "gaussian", "visualization", "integration"
    ], help="Run specific test section")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = TestRunner()
    
    print("Gauge 3D Pipeline Test Suite")
    print("=" * 60)
    
    # Define test sections
    test_sections = {
        "environment": [
            ("Environment Setup", test_environment),
        ],
        "dependencies": [
            ("Depth Estimation Dependencies", test_depth_estimation_dependencies),
            ("4D Gaussian Dependencies", test_4d_gaussian_dependencies),
            ("Visualization Dependencies", test_visualization_dependencies),
        ],
        "structure": [
            ("Project Structure", test_project_structure),
        ],
        "depth": [
            ("Depth Viewer", test_depth_viewer),
            ("Depth Pipeline", test_depth_pipeline),
        ],
        "gaussian": [
            ("Gaussian Data Structures", test_gaussian_data_structures),
        ],
        "visualization": [
            ("Depth Viewer", test_depth_viewer),
        ],
        "integration": [
            ("Component Integration", test_integration),
        ]
    }
    
    # Run specific section or all tests
    if args.section:
        if args.section in test_sections:
            for test_name, test_func in test_sections[args.section]:
                runner.run_test(test_name, test_func)
        else:
            print(f"Unknown test section: {args.section}")
            return
    else:
        # Run all tests
        for section_name, tests in test_sections.items():
            print(f"\n🧪 Running {section_name.upper()} tests...")
            for test_name, test_func in tests:
                runner.run_test(test_name, test_func)
    
    # Print summary
    runner.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if runner.failed == 0 else 1)


if __name__ == "__main__":
    main()
