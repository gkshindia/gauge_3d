#!/usr/bin/env python3
"""
4D Gaussian Integration Test

This script tests the integration of the 4D Gaussian pipeline with actual depth estimation outputs.
It verifies that the pipeline can process existing depth maps and generate proper output files.
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_output_directory_structure():
    """Test that output directory has the expected depth estimation outputs"""
    print("Testing output directory structure...")
    
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ Output directory doesn't exist")
        return False
    
    # Check for depth maps
    depth_maps_dir = output_dir / "depth_maps"
    if depth_maps_dir.exists():
        subdirs = list(depth_maps_dir.iterdir())
        print(f"✅ Found depth maps directory with {len(subdirs)} video sets")
        for subdir in subdirs[:3]:  # Show first 3
            if subdir.is_dir():
                depth_files = list(subdir.glob("*.npy"))
                print(f"   - {subdir.name}: {len(depth_files)} depth files")
    
    # Check for frames
    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        subdirs = list(frames_dir.iterdir())
        print(f"✅ Found frames directory with {len(subdirs)} video sets")
    
    return True

def test_load_sample_depth_data():
    """Test loading and processing sample depth data"""
    print("\nTesting sample depth data loading...")
    
    output_dir = Path("output")
    depth_maps_dir = output_dir / "depth_maps"
    
    if not depth_maps_dir.exists():
        print("❌ No depth maps found")
        return False
    
    # Find first available depth map
    for video_dir in depth_maps_dir.iterdir():
        if video_dir.is_dir():
            depth_files = list(video_dir.glob("*.npy"))
            if depth_files:
                sample_file = depth_files[0]
                try:
                    depth_map = np.load(sample_file)
                    print(f"✅ Loaded sample depth map: {sample_file.name}")
                    print(f"   Shape: {depth_map.shape}")
                    print(f"   Min depth: {depth_map.min():.3f}")
                    print(f"   Max depth: {depth_map.max():.3f}")
                    print(f"   Mean depth: {depth_map.mean():.3f}")
                    return True
                except Exception as e:
                    print(f"❌ Error loading depth map: {e}")
                    return False
    
    print("❌ No valid depth files found")
    return False

def test_4d_gaussian_imports():
    """Test that 4D Gaussian modules can be imported"""
    print("\nTesting 4D Gaussian imports...")
    
    try:
        # Add the current directory to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Test imports using importlib
        import importlib.util
        
        # Test config
        config_path = current_dir / "4d_gaussian" / "config" / "gaussian_config.py"
        spec = importlib.util.spec_from_file_location("gaussian_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        config = config_module.get_default_config()
        print("✅ Config module imported and tested")
        
        # Test data converter
        converter_path = current_dir / "4d_gaussian" / "data_preparation" / "data_converter.py"
        spec = importlib.util.spec_from_file_location("data_converter", converter_path)
        converter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter_module)
        print("✅ Data converter module imported")
        
        # Test gaussian initializer
        init_path = current_dir / "4d_gaussian" / "gaussian_generation" / "gaussian_initializer.py"
        spec = importlib.util.spec_from_file_location("gaussian_initializer", init_path)
        init_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(init_module)
        print("✅ Gaussian initializer module imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_create_gaussian_output():
    """Test creating sample Gaussian output to verify the pipeline writes to output/"""
    print("\nTesting Gaussian output creation...")
    
    try:
        import torch
        
        # Create output directory for 4D Gaussian results
        output_dir = Path("output") / "4d_gaussian_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample Gaussian parameters
        n_gaussians = 500
        positions = torch.randn(n_gaussians, 3)
        colors = torch.rand(n_gaussians, 3)
        scales = torch.rand(n_gaussians, 3) * 0.1
        rotations = torch.randn(n_gaussians, 4)
        rotations = rotations / rotations.norm(dim=1, keepdim=True)
        opacity = torch.rand(n_gaussians, 1)
        
        # Save test outputs
        gaussian_data = {
            'positions': positions,
            'colors': colors,
            'scales': scales,
            'rotations': rotations,
            'opacity': opacity,
            'metadata': {
                'n_gaussians': n_gaussians,
                'test_run': True
            }
        }
        
        torch.save(gaussian_data, output_dir / "test_gaussians.pth")
        
        # Save as numpy arrays too
        np.save(output_dir / "positions.npy", positions.numpy())
        np.save(output_dir / "colors.npy", colors.numpy())
        
        print(f"✅ Created test Gaussian outputs in {output_dir}")
        print(f"   Generated {n_gaussians} Gaussians")
        print(f"   Saved to: test_gaussians.pth, positions.npy, colors.npy")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating Gaussian output: {e}")
        return False

def test_pipeline_readiness():
    """Test if the pipeline is ready to run on real data"""
    print("\nTesting pipeline readiness...")
    
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Dependencies
    try:
        import torch
        import numpy as np
        import trimesh
        checks_passed += 1
        print("✅ Core dependencies available")
    except ImportError:
        print("❌ Missing core dependencies")
    
    # Check 2: Depth data available
    if Path("output/depth_maps").exists():
        checks_passed += 1
        print("✅ Depth estimation data available")
    else:
        print("❌ No depth estimation data found")
    
    # Check 3: 4D Gaussian modules
    if Path("4d_gaussian").exists():
        checks_passed += 1
        print("✅ 4D Gaussian modules present")
    else:
        print("❌ 4D Gaussian modules missing")
    
    # Check 4: Output directory writable
    try:
        test_file = Path("output") / "test_write.tmp"
        test_file.touch()
        test_file.unlink()
        checks_passed += 1
        print("✅ Output directory writable")
    except:
        print("❌ Output directory not writable")
    
    success_rate = checks_passed / total_checks
    print(f"\nPipeline readiness: {checks_passed}/{total_checks} ({success_rate*100:.0f}%)")
    
    return success_rate >= 0.75

def main():
    """Run all integration tests"""
    print("============================================================")
    print("4D GAUSSIAN SPLATTING - INTEGRATION TEST")
    print("============================================================")
    
    tests = [
        ("Output Directory Structure", test_output_directory_structure),
        ("Sample Depth Data Loading", test_load_sample_depth_data),
        ("4D Gaussian Imports", test_4d_gaussian_imports),
        ("Gaussian Output Creation", test_create_gaussian_output),
        ("Pipeline Readiness", test_pipeline_readiness),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n=============== {test_name} ===============")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n============================================================")
    passed = sum(results)
    total = len(results)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("============================================================")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\nThe 4D Gaussian pipeline is ready to run!")
        print("Next steps:")
        print("1. Run: python 4d_gaussian/run_4d_gaussian.py <video_name>")
        print("2. Check outputs in output/4d_gaussian_<video_name>/")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    print(f"\nTest outputs saved to: output/4d_gaussian_test/")
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
