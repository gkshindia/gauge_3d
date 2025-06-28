#!/usr/bin/env python3
"""
Consolidated Test Runner for Gauge 3D Project

This script runs all tests for the Gauge 3D video processing and 
4D Gaussian splatting pipeline. Tests are organized into modules
in the tests/ directory.

Usage:
    python test.py              # Run all tests
    python test.py --module env # Run specific test module
    python test.py --verbose    # Run with verbose output
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import test modules
try:
    from tests.test_environment import test_environment
    from tests.test_dependencies import (
        test_environment as test_env_deps,
        test_depth_estimation_dependencies,
        test_4d_gaussian_dependencies,
        test_visualization_dependencies,
        test_transformer_dependencies
    )
    from tests.test_structure import (
        test_project_structure,
        test_depth_estimation_structure,
        test_4d_gaussian_structure,
        test_viewer_structure,
        test_transformer_structure,
        test_output_structure
    )
    from tests.test_depth import (
        test_depth_pipeline_import,
        test_depth_config,
        test_depth_model_setup,
        test_depth_output_paths,
        test_video_input,
        test_depth_viewer
    )
    from tests.test_gaussian import (
        test_gaussian_pipeline_import,
        test_gaussian_config,
        test_gaussian_dependencies,
        test_gaussian_output_paths,
        test_gaussian_data_format,
        test_gaussian_viewer,
        test_point_cloud_processing
    )
    from tests.test_integration import (
        test_pipeline_integration,
        test_end_to_end_workflow,
        test_data_flow,
        test_viewer_integration,
        test_configuration_consistency,
        test_output_format_compatibility
    )
    from tests.test_transformer import (
        test_transformer_structure,
        test_point_cloud_extraction,
        test_p4transformer_integration,
        test_enhancement_pipeline,
        test_reconstruction_pipeline,
        test_transformer_config,
        test_quality_metrics
    )
    
    TESTS_IMPORTED = True
    
except ImportError as e:
    print(f"Warning: Could not import all test modules: {e}")
    print("Some tests may not be available.")
    TESTS_IMPORTED = False


def run_environment_tests():
    """Run environment and dependency tests"""
    print("\n" + "="*50)
    print("ENVIRONMENT & DEPENDENCY TESTS")
    print("="*50)
    
    results = []
    
    if TESTS_IMPORTED:
        # Environment setup
        results.append(("Environment Setup", test_environment()))
        results.append(("Environment Dependencies", test_env_deps()))
        
        # Component dependencies
        results.append(("Depth Dependencies", test_depth_estimation_dependencies()))
        results.append(("Gaussian Dependencies", test_4d_gaussian_dependencies()))
        results.append(("Visualization Dependencies", test_visualization_dependencies()))
        results.append(("Transformer Dependencies", test_transformer_dependencies()))
    else:
        print("❌ Test modules not available")
        results.append(("Test Import", False))
    
    return results


def run_structure_tests():
    """Run project structure tests"""
    print("\n" + "="*50)
    print("PROJECT STRUCTURE TESTS")
    print("="*50)
    
    results = []
    
    if TESTS_IMPORTED:
        results.append(("Project Structure", test_project_structure()))
        results.append(("Depth Structure", test_depth_estimation_structure()))
        results.append(("Gaussian Structure", test_4d_gaussian_structure()))
        results.append(("Viewer Structure", test_viewer_structure()))
        results.append(("Transformer Structure", test_transformer_structure()))
        results.append(("Output Structure", test_output_structure()))
    else:
        print("❌ Test modules not available")
        results.append(("Test Import", False))
    
    return results


def run_depth_tests():
    """Run depth estimation tests"""
    print("\n" + "="*50)
    print("DEPTH ESTIMATION TESTS")
    print("="*50)
    
    results = []
    
    if TESTS_IMPORTED:
        results.append(("Depth Pipeline Import", test_depth_pipeline_import()))
        results.append(("Depth Configuration", test_depth_config()))
        results.append(("Depth Model Setup", test_depth_model_setup()))
        results.append(("Depth Output Paths", test_depth_output_paths()))
        results.append(("Video Input", test_video_input()))
        results.append(("Depth Viewer", test_depth_viewer()))
    else:
        print("❌ Test modules not available")
        results.append(("Test Import", False))
    
    return results


def run_gaussian_tests():
    """Run 4D Gaussian tests"""
    print("\n" + "="*50)
    print("4D GAUSSIAN TESTS")
    print("="*50)
    
    results = []
    
    if TESTS_IMPORTED:
        results.append(("Gaussian Pipeline Import", test_gaussian_pipeline_import()))
        results.append(("Gaussian Configuration", test_gaussian_config()))
        results.append(("Gaussian Dependencies", test_gaussian_dependencies()))
        results.append(("Gaussian Output Paths", test_gaussian_output_paths()))
        results.append(("Gaussian Data Format", test_gaussian_data_format()))
        results.append(("Gaussian Viewer", test_gaussian_viewer()))
        results.append(("Point Cloud Processing", test_point_cloud_processing()))
    else:
        print("❌ Test modules not available")
        results.append(("Test Import", False))
    
    return results


def run_integration_tests():
    """Run integration tests"""
    print("\n" + "="*50)
    print("INTEGRATION TESTS")
    print("="*50)
    
    results = []
    
    if TESTS_IMPORTED:
        results.append(("Pipeline Integration", test_pipeline_integration()))
        results.append(("End-to-End Workflow", test_end_to_end_workflow()))
        results.append(("Data Flow", test_data_flow()))
        results.append(("Viewer Integration", test_viewer_integration()))
        results.append(("Configuration Consistency", test_configuration_consistency()))
        results.append(("Output Format Compatibility", test_output_format_compatibility()))
    else:
        print("❌ Test modules not available")
        results.append(("Test Import", False))
    
    return results


def run_transformer_tests():
    """Run transformer enhancement tests (Phase 3)"""
    print("\n" + "="*50)
    print("TRANSFORMER ENHANCEMENT TESTS (Phase 3)")
    print("="*50)
    
    results = []
    
    if TESTS_IMPORTED:
        results.append(("Transformer Structure", test_transformer_structure()))
        results.append(("Point Cloud Extraction", test_point_cloud_extraction()))
        results.append(("P4Transformer Integration", test_p4transformer_integration()))
        results.append(("Enhancement Pipeline", test_enhancement_pipeline()))
        results.append(("Reconstruction Pipeline", test_reconstruction_pipeline()))
        results.append(("Transformer Configuration", test_transformer_config()))
        results.append(("Quality Metrics", test_quality_metrics()))
    else:
        print("❌ Test modules not available")
        results.append(("Test Import", False))
    
    return results


def run_all_tests():
    """Run all test suites"""
    all_results = []
    
    all_results.extend(run_environment_tests())
    all_results.extend(run_structure_tests())
    all_results.extend(run_depth_tests())
    all_results.extend(run_gaussian_tests())
    all_results.extend(run_integration_tests())
    all_results.extend(run_transformer_tests())
    
    return all_results


def print_summary(results):
    """Print test results summary"""
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    total = len(results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    
    if failed > 0:
        print("\nFailed Tests:")
        for test_name, result in results:
            if not result:
                print(f"  ❌ {test_name}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return failed == 0


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run Gauge 3D tests")
    parser.add_argument(
        "--module", 
        choices=["env", "structure", "depth", "gaussian", "integration", "transformer"],
        help="Run specific test module"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("Gauge 3D Test Suite")
    print("="*50)
    
    if args.module == "env":
        results = run_environment_tests()
    elif args.module == "structure":
        results = run_structure_tests()
    elif args.module == "depth":
        results = run_depth_tests()
    elif args.module == "gaussian":
        results = run_gaussian_tests()
    elif args.module == "integration":
        results = run_integration_tests()
    elif args.module == "transformer":
        results = run_transformer_tests()
    else:
        results = run_all_tests()
    
    success = print_summary(results)
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
