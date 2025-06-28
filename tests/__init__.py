"""
Test Suite for Gauge 3D Project

This package contains all tests for the Gauge 3D video processing and 
4D Gaussian splatting pipeline.

Test Modules:
- test_environment: Environment and dependency validation
- test_dependencies: Dependency validation for all components
- test_structure: Project structure validation
- test_depth: Depth estimation pipeline tests
- test_gaussian: 4D Gaussian splatting tests
- test_integration: Integration tests across components
- test_transformer: Transformer enhancement tests (Phase 3)
"""

import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import test modules
from . import test_environment
from . import test_dependencies
from . import test_structure
from . import test_depth
from . import test_gaussian
from . import test_integration
from . import test_transformer

__all__ = [
    'test_environment',
    'test_dependencies', 
    'test_structure',
    'test_depth',
    'test_gaussian',
    'test_integration',
    'test_transformer'
]
