#!/usr/bin/env python3
"""
Test Phase 2 Setup and Basic Functionality

This script tests the 4D Gaussian Splatting pipeline setup without
requiring the full GPU-based dependencies.
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all basic imports work"""
    print("Testing imports...")
    
    try:
        # Test config
        import importlib.util
        spec = importlib.util.spec_from_file_location("gaussian_config", "4d_gaussian/config/gaussian_config.py")
        gaussian_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gaussian_config)
        config = gaussian_config.get_default_config()
        config = get_default_config()
        print("✅ Config import successful")
        
        # Test basic dependencies
        import torch
        import numpy as np
        import trimesh
        import plyfile
        import roma
        import kornia
        print("✅ Basic dependencies import successful")
        
        # Test data converter (without Open3D parts)
        import sys
        sys.path.append('4d_gaussian')
        from data_preparation.data_converter import DepthToGaussianConverter
        print("✅ Data converter import successful")
        
        # Test Gaussian initializer
        from gaussian_generation.gaussian_initializer import GaussianInitializer
        print("✅ Gaussian initializer import successful")
        
        # Test visualization
        from utils.visualization import GaussianVisualizer
        print("✅ Visualization import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_functionality():
    """Test configuration functionality"""
    print("\nTesting configuration...")
    
    try:
        from 4d_gaussian.config.gaussian_config import get_default_config, get_data_config
        
        # Test default config
        config = get_default_config()
        assert config.max_gaussians > 0
        assert config.temporal_window > 0
        print("✅ Default config validation passed")
        
        # Test data config
        data_config = get_data_config()
        assert data_config.frames_dir
        assert data_config.depth_maps_dir
        print("✅ Data config validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_gaussian_initialization():
    """Test Gaussian initialization with synthetic data"""
    print("\nTesting Gaussian initialization...")
    
    try:
        from 4d_gaussian.gaussian_generation.gaussian_initializer import GaussianInitializer
        
        # Create synthetic point cloud
        n_points = 1000
        positions = np.random.randn(n_points, 3).astype(np.float32)
        colors = np.random.rand(n_points, 3).astype(np.float32)
        point_cloud = np.concatenate([positions, colors], axis=1)
        
        # Initialize Gaussians
        config = {"device": "cpu", "max_gaussians": 2000}
        initializer = GaussianInitializer(config)
        
        gaussians = initializer.initialize_gaussians_from_pointcloud(point_cloud)
        
        # Validate results
        assert "positions" in gaussians
        assert "colors" in gaussians
        assert "scales" in gaussians
        assert "rotations" in gaussians
        assert "opacity" in gaussians
        
        assert len(gaussians["positions"]) == n_points
        print(f"✅ Gaussian initialization successful: {len(gaussians['positions'])} Gaussians")
        
        return True
        
    except Exception as e:
        print(f"❌ Gaussian initialization test failed: {e}")
        return False

def test_depth_conversion():
    """Test depth map to point cloud conversion"""
    print("\nTesting depth conversion...")
    
    try:
        from 4d_gaussian.data_preparation.data_converter import DepthToGaussianConverter
        
        # Create synthetic RGB and depth data
        height, width = 480, 640
        rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        depth_map = np.random.uniform(0.5, 5.0, (height, width)).astype(np.float32)
        
        # Create converter
        converter = DepthToGaussianConverter("", "", "test_output")
        
        # Test depth to point cloud conversion
        intrinsics = {
            "fx": 500.0, "fy": 500.0,
            "cx": width/2, "cy": height/2,
            "width": width, "height": height
        }
        
        point_cloud = converter.depth_to_point_cloud(rgb_image, depth_map, intrinsics)
        
        # Validate
        assert point_cloud.shape[1] == 6  # xyz + rgb
        assert len(point_cloud) > 0
        print(f"✅ Depth conversion successful: {len(point_cloud)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Depth conversion test failed: {e}")
        return False

def test_visualization():
    """Test visualization functionality"""
    print("\nTesting visualization...")
    
    try:
        from 4d_gaussian.utils.visualization import GaussianVisualizer
        
        # Create synthetic Gaussian data
        n_gaussians = 500
        gaussians = {
            "positions": np.random.randn(n_gaussians, 3).astype(np.float32),
            "colors": np.random.rand(n_gaussians, 3).astype(np.float32),
            "scales": np.random.uniform(0.01, 0.1, (n_gaussians, 3)).astype(np.float32),
            "opacity": np.random.uniform(-2, 2, (n_gaussians, 1)).astype(np.float32),
        }
        
        # Test visualization
        visualizer = GaussianVisualizer("test_output/viz")
        
        # Test statistics plotting (save to file to avoid display issues)
        test_plot_path = "test_output/test_gaussian_stats.png"
        os.makedirs("test_output", exist_ok=True)
        
        visualizer.plot_gaussian_statistics(gaussians, test_plot_path)
        
        if os.path.exists(test_plot_path):
            print("✅ Visualization test successful")
            # Clean up
            os.remove(test_plot_path)
            return True
        else:
            print("❌ Visualization test failed - no output file")
            return False
            
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("4D GAUSSIAN SPLATTING - PHASE 2 SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config_functionality),
        ("Gaussian Initialization", test_gaussian_initialization),
        ("Depth Conversion", test_depth_conversion),
        ("Visualization", test_visualization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Phase 2 setup is working correctly.")
        print("\nNext steps:")
        print("1. Run environment setup: python 4d_gaussian/setup/environment_setup.py")
        print("2. Test with real data: python 4d_gaussian/run_phase2.py your_video_name")
        return True
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
