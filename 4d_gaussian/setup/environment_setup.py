"""
Environment Setup for 4D Gaussian Splatting

This module handles the setup and configuration of the environment required
for 4D Gaussian Splatting, including dependencies, GPU setup, and model downloads.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import platform

class GaussianEnvironmentSetup:
    """Setup and configure environment for 4D Gaussian Splatting"""
    
    def __init__(self, config_dir: str = "4d_gaussian/config"):
        self.config_dir = Path(config_dir)
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Environment markers
        self.markers_dir = self.config_dir / "markers"
        self.markers_dir.mkdir(parents=True, exist_ok=True)
        
        # Required dependencies
        self.required_packages = [
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "pytorch3d",
            "simple-knn",
            "diff-gaussian-rasterization",
            "open3d>=0.17.0",
            "plyfile>=0.7.4",
            "roma>=1.2.0",
            "kornia>=0.6.12",
            "nerfstudio>=0.3.0",
            "trimesh>=3.21.0",
            "pymeshlab>=2022.2",
        ]
        
    def setup_logging(self):
        """Configure logging for the setup process"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('4d_gaussian_setup.log')
            ]
        )
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements for 4D Gaussian Splatting"""
        self.logger.info("Checking system requirements...")
        
        requirements = {
            "python_version": sys.version_info[:2],
            "platform": platform.system(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_memory": [],
            "cpu_count": os.cpu_count(),
            "ram_gb": None
        }
        
        # Check CUDA
        if torch.cuda.is_available():
            requirements["cuda_version"] = torch.version.cuda
            requirements["gpu_count"] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                gpu_properties = torch.cuda.get_device_properties(i)
                requirements["gpu_memory"].append({
                    "device": i,
                    "name": gpu_properties.name,
                    "total_memory_gb": gpu_properties.total_memory / 1024**3,
                    "multi_processor_count": gpu_properties.multi_processor_count
                })
        
        # Check RAM
        try:
            import psutil
            requirements["ram_gb"] = psutil.virtual_memory().total / 1024**3
        except ImportError:
            self.logger.warning("psutil not available, cannot check RAM")
        
        self.logger.info(f"System requirements check complete: {requirements}")
        return requirements
    
    def validate_requirements(self, requirements: Dict[str, Any]) -> bool:
        """Validate that system meets minimum requirements"""
        self.logger.info("Validating system requirements...")
        
        issues = []
        
        # Check Python version
        if requirements["python_version"] < (3, 8):
            issues.append("Python 3.8+ required")
        
        # Check CUDA
        if not requirements["cuda_available"]:
            issues.append("CUDA not available - GPU acceleration strongly recommended")
        elif requirements["gpu_count"] == 0:
            issues.append("No CUDA devices found")
        else:
            # Check GPU memory
            min_gpu_memory = 6.0  # GB
            has_sufficient_memory = any(
                gpu["total_memory_gb"] >= min_gpu_memory 
                for gpu in requirements["gpu_memory"]
            )
            if not has_sufficient_memory:
                issues.append(f"At least {min_gpu_memory}GB GPU memory recommended")
        
        # Check RAM
        if requirements["ram_gb"] and requirements["ram_gb"] < 16:
            issues.append("At least 16GB RAM recommended")
        
        if issues:
            self.logger.warning(f"System validation issues: {issues}")
            return False
        
        self.logger.info("System requirements validation passed")
        return True
    
    def install_pytorch3d(self) -> bool:
        """Install PyTorch3D with proper CUDA support"""
        self.logger.info("Installing PyTorch3D...")
        
        try:
            # Check if already installed
            import pytorch3d
            self.logger.info("PyTorch3D already installed")
            return True
        except ImportError:
            pass
        
        # Install PyTorch3D
        try:
            if torch.cuda.is_available():
                # Install with CUDA support
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "pytorch3d", "-f", 
                    "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt200/download.html"
                ], check=True)
            else:
                # CPU-only installation
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "pytorch3d"
                ], check=True)
            
            self.logger.info("PyTorch3D installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install PyTorch3D: {e}")
            return False
    
    def install_gaussian_rasterization(self) -> bool:
        """Install differential Gaussian rasterization"""
        self.logger.info("Installing differential Gaussian rasterization...")
        
        try:
            # Check if already installed
            import diff_gaussian_rasterization
            self.logger.info("Gaussian rasterization already installed")
            return True
        except ImportError:
            pass
        
        try:
            # Clone and install from source
            subprocess.run([
                "git", "clone", 
                "https://github.com/graphdeco-inria/diff-gaussian-rasterization",
                "/tmp/diff-gaussian-rasterization"
            ], check=True)
            
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "/tmp/diff-gaussian-rasterization"
            ], check=True)
            
            self.logger.info("Gaussian rasterization installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install Gaussian rasterization: {e}")
            return False
    
    def install_simple_knn(self) -> bool:
        """Install simple-knn for nearest neighbor operations"""
        self.logger.info("Installing simple-knn...")
        
        try:
            import simple_knn
            self.logger.info("simple-knn already installed")
            return True
        except ImportError:
            pass
        
        try:
            subprocess.run([
                "git", "clone", 
                "https://gitlab.inria.fr/bkerbl/simple-knn.git",
                "/tmp/simple-knn"
            ], check=True)
            
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "/tmp/simple-knn"
            ], check=True)
            
            self.logger.info("simple-knn installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install simple-knn: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install all required dependencies"""
        self.logger.info("Installing 4D Gaussian dependencies...")
        
        # Install PyTorch3D first
        if not self.install_pytorch3d():
            return False
        
        # Install Gaussian rasterization
        if not self.install_gaussian_rasterization():
            return False
        
        # Install simple-knn
        if not self.install_simple_knn():
            return False
        
        # Install other packages
        try:
            other_packages = [
                "open3d>=0.17.0",
                "plyfile>=0.7.4", 
                "roma>=1.2.0",
                "kornia>=0.6.12",
                "trimesh>=3.21.0",
                "pymeshlab>=2022.2"
            ]
            
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + other_packages, check=True)
            
            self.logger.info("All dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that all components are properly installed"""
        self.logger.info("Verifying 4D Gaussian installation...")
        
        required_imports = [
            "torch",
            "torchvision", 
            "pytorch3d",
            "diff_gaussian_rasterization",
            "simple_knn",
            "open3d",
            "plyfile",
            "roma",
            "kornia",
            "trimesh"
        ]
        
        failed_imports = []
        for module in required_imports:
            try:
                __import__(module)
                self.logger.info(f"✓ {module} imported successfully")
            except ImportError as e:
                self.logger.error(f"✗ Failed to import {module}: {e}")
                failed_imports.append(module)
        
        if failed_imports:
            self.logger.error(f"Failed imports: {failed_imports}")
            return False
        
        # Create marker file
        marker_file = self.markers_dir / "4d_gaussian_setup_complete.marker"
        marker_file.write_text("4D Gaussian setup completed successfully")
        
        self.logger.info("4D Gaussian installation verification completed successfully")
        return True
    
    def setup_environment(self) -> bool:
        """Complete environment setup process"""
        self.logger.info("Starting 4D Gaussian environment setup...")
        
        # Check if already set up
        marker_file = self.markers_dir / "4d_gaussian_setup_complete.marker"
        if marker_file.exists():
            self.logger.info("4D Gaussian environment already set up")
            return True
        
        # Check system requirements
        requirements = self.check_system_requirements()
        if not self.validate_requirements(requirements):
            self.logger.warning("System requirements validation failed, continuing anyway...")
        
        # Install dependencies
        if not self.install_dependencies():
            self.logger.error("Failed to install dependencies")
            return False
        
        # Verify installation
        if not self.verify_installation():
            self.logger.error("Installation verification failed")
            return False
        
        self.logger.info("4D Gaussian environment setup completed successfully")
        return True

def main():
    """Main entry point for environment setup"""
    setup = GaussianEnvironmentSetup()
    success = setup.setup_environment()
    
    if success:
        print("✅ 4D Gaussian environment setup completed successfully")
        sys.exit(0)
    else:
        print("❌ 4D Gaussian environment setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
