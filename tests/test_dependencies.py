#!/usr/bin/env python3
"""
Test module for dependency validation
"""

import sys
import torch
import numpy as np
import cv2
import matplotlib
import logging

logger = logging.getLogger(__name__)


def test_environment():
    """Test environment setup and basic dependencies"""
    print("Testing environment setup...")
    
    try:
        # Test Python version
        python_version = sys.version_info
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Test basic imports
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ NumPy {np.__version__}")
        print(f"✅ OpenCV {cv2.__version__}")
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


def test_transformer_dependencies():
    """Test transformer enhancement dependencies"""
    print("Testing transformer dependencies...")
    
    try:
        import torch
        import transformers
        import numpy as np
        
        # Test specific transformer requirements
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ Transformers {transformers.__version__}")
        print(f"✅ NumPy {np.__version__}")
        
        # Test point cloud processing libraries
        try:
            import open3d as o3d
            print(f"✅ Open3D {o3d.__version__}")
        except ImportError:
            print("❌ Open3D required for point cloud processing")
            return False
            
        try:
            import trimesh
            print(f"✅ Trimesh {trimesh.__version__}")
        except ImportError:
            print("⚠️  Trimesh not available (optional)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Transformer dependencies failed: {e}")
        return False
