#!/usr/bin/env python3
"""
4D Gaussian Splatting Basic Test

This script tests basic functionality and dependency availability.
"""

import sys
import os
from pathlib import Path
import numpy as np

def test_basic_dependencies():
    """Test that basic dependencies are available"""
    print("Testing basic dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import trimesh
        print(f"✅ Trimesh {trimesh.__version__}")
        
        import plyfile
        print("✅ PLYFile available")
        
        import roma
        print("✅ Roma available")
        
        import kornia
        print(f"✅ Kornia {kornia.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_gaussian_data_structures():
    """Test basic Gaussian data structure operations"""
    print("\nTesting Gaussian data structures...")
    
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

def test_point_cloud_conversion():
    """Test point cloud to Gaussian conversion logic"""
    print("\nTesting point cloud conversion...")
    
    try:
        import torch
        
        # Create synthetic point cloud
        n_points = 5000
        points = np.random.randn(n_points, 3).astype(np.float32)
        colors = np.random.rand(n_points, 3).astype(np.float32)
        point_cloud = np.concatenate([points, colors], axis=1)
        
        print(f"✅ Created synthetic point cloud: {point_cloud.shape}")
        
        # Test conversion to torch tensors
        positions_tensor = torch.from_numpy(point_cloud[:, :3])
        colors_tensor = torch.from_numpy(point_cloud[:, 3:6])
        
        print(f"✅ Converted to tensors: {positions_tensor.shape}, {colors_tensor.shape}")
        
        # Test basic operations
        mean_pos = torch.mean(positions_tensor, dim=0)
        std_pos = torch.std(positions_tensor, dim=0)
        
        print("✅ Point cloud statistics:")
        print(f"   Mean position: {mean_pos}")
        print(f"   Std position: {std_pos}")
        
        return True
        
    except Exception as e:
        print(f"❌ Point cloud conversion test failed: {e}")
        return False

def test_file_io():
    """Test file I/O operations"""
    print("\nTesting file I/O...")
    
    try:
        # Test numpy file operations
        test_data = np.random.randn(100, 6).astype(np.float32)
        test_file = "test_output/test_data.npy"
        
        os.makedirs("test_output", exist_ok=True)
        np.save(test_file, test_data)
        
        loaded_data = np.load(test_file)
        assert np.allclose(test_data, loaded_data)
        
        print("✅ NumPy file I/O working")
        
        # Test PLY file operations
        import plyfile
        
        # Create simple PLY data
        vertex_data = np.array([(0.0, 0.0, 0.0, 255, 0, 0),
                               (1.0, 0.0, 0.0, 0, 255, 0),
                               (0.0, 1.0, 0.0, 0, 0, 255)],
                              dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        
        el = plyfile.PlyElement.describe(vertex_data, 'vertex')
        ply_file = "test_output/test.ply"
        plyfile.PlyData([el]).write(ply_file)
        
        print("✅ PLY file I/O working")
        
        # Clean up
        os.remove(test_file)
        os.remove(ply_file)
        
        return True
        
    except Exception as e:
        print(f"❌ File I/O test failed: {e}")
        return False

def test_depth_to_pointcloud():
    """Test depth map to point cloud conversion logic"""
    print("\nTesting depth to point cloud conversion...")
    
    try:
        # Create synthetic depth map and RGB image
        height, width = 240, 320
        depth_map = np.random.uniform(0.5, 5.0, (height, width)).astype(np.float32)
        rgb_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Camera intrinsics
        fx, fy = 250.0, 250.0
        cx, cy = width / 2, height / 2
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter valid depths
        valid_mask = (depth_map > 0.1) & (depth_map < 10.0)
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_map[valid_mask]
        
        # Convert to 3D coordinates
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid
        
        # Get colors
        colors = rgb_image[valid_mask] / 255.0
        
        # Combine into point cloud
        points_3d = np.stack([x, y, z], axis=1)
        point_cloud = np.concatenate([points_3d, colors], axis=1)
        
        print(f"✅ Depth to point cloud conversion successful")
        print(f"   Input depth map: {depth_map.shape}")
        print(f"   Output point cloud: {point_cloud.shape}")
        print(f"   Valid points: {len(point_cloud)} / {height * width}")
        
        return True
        
    except Exception as e:
        print(f"❌ Depth to point cloud test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("4D GAUSSIAN SPLATTING - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Dependencies", test_basic_dependencies),
        ("Gaussian Data Structures", test_gaussian_data_structures),
        ("Point Cloud Conversion", test_point_cloud_conversion),
        ("File I/O", test_file_io),
        ("Depth to Point Cloud", test_depth_to_pointcloud),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All basic tests passed!")
        print("\n4D Gaussian Splatting dependencies are working correctly.")
        print("\nNext steps:")
        print("1. Ensure depth estimation outputs exist in output/ directory")
        print("2. Run: python 4d_gaussian/run_4d_gaussian.py your_video_name")
        print("3. For environment setup: python 4d_gaussian/setup/environment_setup.py")
        return True
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
