#!/usr/bin/env python3
"""
Test module for project structure validation
"""

import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def test_project_structure():
    """Test that all required project directories and files exist"""
    print("Testing project structure...")
    
    project_root = Path(__file__).parent.parent
    
    # Required directories
    required_dirs = [
        "src",
        "viewer", 
        "depth_estimation",
        "4d_gaussian",
        "tests",
        "output",
        "vinput"
    ]
    
    # Required files
    required_files = [
        "README.md",
        "pyproject.toml",
        ".gitignore",
        ".env.template"
    ]
    
    all_passed = True
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ Directory: {dir_name}/")
        else:
            print(f"❌ Missing directory: {dir_name}/")
            all_passed = False
    
    # Check files
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists() and file_path.is_file():
            print(f"✅ File: {file_name}")
        else:
            print(f"❌ Missing file: {file_name}")
            all_passed = False
    
    return all_passed


def test_depth_estimation_structure():
    """Test depth estimation module structure"""
    print("Testing depth estimation structure...")
    
    project_root = Path(__file__).parent.parent
    depth_dir = project_root / "depth_estimation"
    
    required_files = [
        "main_pipeline.py",
        "depth_pipeline.py", 
        "video_depth_processor.py",
        "output_manager.py",
        "setup_environment.py",
        "README.md"
    ]
    
    required_dirs = [
        "config",
        "utils"
    ]
    
    all_passed = True
    
    for file_name in required_files:
        file_path = depth_dir / file_name
        if file_path.exists():
            print(f"✅ Depth file: {file_name}")
        else:
            print(f"❌ Missing depth file: {file_name}")
            all_passed = False
    
    for dir_name in required_dirs:
        dir_path = depth_dir / dir_name
        if dir_path.exists():
            print(f"✅ Depth directory: {dir_name}/")
        else:
            print(f"❌ Missing depth directory: {dir_name}/")
            all_passed = False
    
    return all_passed


def test_4d_gaussian_structure():
    """Test 4D Gaussian module structure"""
    print("Testing 4D Gaussian structure...")
    
    project_root = Path(__file__).parent.parent
    gaussian_dir = project_root / "4d_gaussian"
    
    required_files = [
        "__init__.py",
        "run_4d_gaussian.py",
        "run_phase2.py",
        "README.md"
    ]
    
    required_dirs = [
        "config",
        "data_preparation",
        "gaussian_generation",
        "optimization",
        "rendering",
        "setup",
        "utils"
    ]
    
    all_passed = True
    
    for file_name in required_files:
        file_path = gaussian_dir / file_name
        if file_path.exists():
            print(f"✅ Gaussian file: {file_name}")
        else:
            print(f"❌ Missing Gaussian file: {file_name}")
            all_passed = False
    
    for dir_name in required_dirs:
        dir_path = gaussian_dir / dir_name
        if dir_path.exists():
            print(f"✅ Gaussian directory: {dir_name}/")
        else:
            print(f"❌ Missing Gaussian directory: {dir_name}/")
            all_passed = False
    
    return all_passed


def test_viewer_structure():
    """Test viewer module structure"""
    print("Testing viewer structure...")
    
    project_root = Path(__file__).parent.parent
    viewer_dir = project_root / "viewer"
    
    required_files = [
        "__init__.py",
        "depth_viewer.py",
        "gaussian_viewer.py", 
        "pipeline_viewer.py",
        "point_cloud_viewer.py",
        "stats_viewer.py",
        "README.md"
    ]
    
    all_passed = True
    
    for file_name in required_files:
        file_path = viewer_dir / file_name
        if file_path.exists():
            print(f"✅ Viewer file: {file_name}")
        else:
            print(f"❌ Missing viewer file: {file_name}")
            all_passed = False
    
    return all_passed


def test_transformer_structure():
    """Test transformer module structure (Phase 3)"""
    print("Testing transformer structure...")
    
    project_root = Path(__file__).parent.parent
    transformer_dir = project_root / "transformer"
    
    # These will be created in Phase 3
    required_files = [
        "__init__.py",
        "point_cloud_extractor.py",
        "p4transformer_integration.py",
        "enhancement_pipeline.py",
        "reconstruction_pipeline.py",
        "README.md"
    ]
    
    required_dirs = [
        "config",
        "models",
        "utils"
    ]
    
    all_passed = True
    
    if not transformer_dir.exists():
        print("⚠️  Transformer directory not yet created (Phase 3)")
        return True  # Not an error yet
    
    for file_name in required_files:
        file_path = transformer_dir / file_name
        if file_path.exists():
            print(f"✅ Transformer file: {file_name}")
        else:
            print(f"❌ Missing transformer file: {file_name}")
            all_passed = False
    
    for dir_name in required_dirs:
        dir_path = transformer_dir / dir_name
        if dir_path.exists():
            print(f"✅ Transformer directory: {dir_name}/")
        else:
            print(f"❌ Missing transformer directory: {dir_name}/")
            all_passed = False
    
    return all_passed


def test_output_structure():
    """Test output directory structure"""
    print("Testing output structure...")
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output"
    
    if not output_dir.exists():
        print("⚠️  Output directory will be created when needed")
        return True
    
    # Check for expected subdirectories
    expected_subdirs = [
        "depth_maps",
        "depth_previews", 
        "depth_stats",
        "frames",
        "gaussian_reconstruction"
    ]
    
    all_passed = True
    
    for subdir in expected_subdirs:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            print(f"✅ Output subdirectory: {subdir}/")
        else:
            print(f"⚠️  Output subdirectory not created yet: {subdir}/")
    
    return all_passed
