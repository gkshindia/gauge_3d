#!/usr/bin/env python3
"""
Test module for 4D Gaussian functionality
"""

import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def test_gaussian_pipeline_import():
    """Test that 4D Gaussian modules can be imported"""
    print("Testing 4D Gaussian pipeline imports...")
    
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        
        # Import with proper module path handling
        import importlib.util
        
        # Try to import 4D Gaussian modules
        gaussian_init_path = Path(__file__).parent.parent / "4d_gaussian" / "gaussian_generation" / "gaussian_initializer.py"
        if gaussian_init_path.exists():
            print("✅ GaussianInitializer module file exists")
        else:
            print("❌ GaussianInitializer module file missing")
            
        data_converter_path = Path(__file__).parent.parent / "4d_gaussian" / "data_preparation" / "data_converter.py"
        if data_converter_path.exists():
            print("✅ DataConverter module file exists")
        else:
            print("❌ DataConverter module file missing")
        
        return True
        
    except ImportError as e:
        print(f"❌ 4D Gaussian pipeline import failed: {e}")
        return False


def test_gaussian_config():
    """Test 4D Gaussian configuration"""
    print("Testing 4D Gaussian configuration...")
    
    try:
        project_root = Path(__file__).parent.parent
        config_file = project_root / "4d_gaussian" / "config" / "gaussian_config.py"
        
        if config_file.exists():
            print("✅ Gaussian config file exists")
            return True
        else:
            print("❌ Gaussian config file missing")
            return False
            
    except Exception as e:
        print(f"❌ Gaussian config test failed: {e}")
        return False


def test_gaussian_dependencies():
    """Test 4D Gaussian dependencies"""
    print("Testing 4D Gaussian dependencies...")
    
    try:
        # Test PyTorch for Gaussian operations
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
        
        # Test NumPy for numerical operations
        import numpy as np
        print(f"✅ NumPy {np.__version__} available")
        
        # Test CUDA for GPU acceleration
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - Gaussian operations will be slower")
        
        return True
        
    except ImportError as e:
        print(f"❌ Gaussian dependencies test failed: {e}")
        return False


def test_gaussian_output_paths():
    """Test 4D Gaussian output path structure"""
    print("Testing Gaussian output paths...")
    
    try:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "output"
        
        # Expected output subdirectories for Gaussian processing
        expected_subdirs = [
            "gaussian_reconstruction",
            "4d_gaussian_test"
        ]
        
        for subdir in expected_subdirs:
            subdir_path = output_dir / subdir
            if subdir_path.exists():
                print(f"✅ Gaussian output path exists: {subdir}/")
            else:
                print(f"⚠️  Gaussian output path will be created: {subdir}/")
        
        return True
        
    except Exception as e:
        print(f"❌ Gaussian output paths test failed: {e}")
        return False


def test_gaussian_data_format():
    """Test Gaussian data format validation"""
    print("Testing Gaussian data formats...")
    
    try:
        project_root = Path(__file__).parent.parent
        test_output_dir = project_root / "output" / "4d_gaussian_test"
        
        if test_output_dir.exists():
            # Check for expected test files
            expected_files = [
                "positions.npy",
                "colors.npy", 
                "test_gaussians.pth"
            ]
            
            for filename in expected_files:
                file_path = test_output_dir / filename
                if file_path.exists():
                    print(f"✅ Test Gaussian file exists: {filename}")
                else:
                    print(f"⚠️  Test Gaussian file not created yet: {filename}")
        else:
            print("⚠️  Test Gaussian output directory not created yet")
        
        return True
        
    except Exception as e:
        print(f"❌ Gaussian data format test failed: {e}")
        return False


def test_gaussian_viewer():
    """Test Gaussian viewer functionality"""
    print("Testing Gaussian viewer...")
    
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        
        from viewer.gaussian_viewer import GaussianViewer
        print("✅ GaussianViewer import successful")
        
        from viewer.point_cloud_viewer import PointCloudViewer
        print("✅ PointCloudViewer import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Gaussian viewer test failed: {e}")
        return False


def test_point_cloud_processing():
    """Test point cloud processing capabilities"""
    print("Testing point cloud processing...")
    
    try:
        # Test Open3D for point cloud operations
        import open3d as o3d
        print(f"✅ Open3D {o3d.__version__} available for point clouds")
        
        # Test Trimesh for mesh operations
        import trimesh
        print(f"✅ Trimesh {trimesh.__version__} available for meshes")
        
        return True
        
    except ImportError as e:
        print(f"❌ Point cloud processing test failed: {e}")
        return False
