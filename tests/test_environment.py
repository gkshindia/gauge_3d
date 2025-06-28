"""
Environment and basic dependency tests
"""

import sys
import torch
import numpy as np
import cv2
import matplotlib


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
