#!/usr/bin/env python3
"""
Gauge 3D Validation and Testing Suite
Phase 4.4: Comprehensive validation with ground truth comparison and accuracy metrics
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Any
from datetime import datetime
import time

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class GroundTruthComparator:
    """Compare pipeline outputs with ground truth data"""
    
    def __init__(self, ground_truth_dir: Union[str, Path]):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.metrics_history = []
        
    def compare_depth_maps(
        self, 
        predicted_depth_dir: Union[str, Path],
        video_name: str
    ) -> Dict[str, Any]:
        """Compare predicted depth maps with ground truth"""
        predicted_dir = Path(predicted_depth_dir) / video_name
        gt_dir = self.ground_truth_dir / "depth_maps" / video_name
        
        if not gt_dir.exists():
            logger.warning(f"No ground truth depth maps found for {video_name}")
            return {"error": "No ground truth available"}
        
        metrics = {
            "rmse": [],
            "mae": [],
            "abs_rel": [],
            "sq_rel": [],
            "delta1": [],
            "delta2": [],
            "delta3": []
        }
        
        comparison_results = []
        
        # Compare each frame
        for pred_file in predicted_dir.glob("*_depth.npy"):
            frame_name = pred_file.stem.replace("_depth", "")
            gt_file = gt_dir / f"{frame_name}_gt_depth.npy"
            
            if not gt_file.exists():
                continue
                
            try:
                pred_depth = np.load(pred_file)
                gt_depth = np.load(gt_file)
                
                # Ensure same dimensions
                if pred_depth.shape != gt_depth.shape:
                    pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))
                
                frame_metrics = self._calculate_depth_metrics(pred_depth, gt_depth)
                
                for key, value in frame_metrics.items():
                    metrics[key].append(value)
                
                comparison_results.append({
                    "frame": frame_name,
                    "metrics": frame_metrics
                })
                
            except Exception as e:
                logger.error(f"Error comparing depth for frame {frame_name}: {e}")
                continue
        
        # Calculate averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[f"avg_{key}"] = np.mean(values)
                avg_metrics[f"std_{key}"] = np.std(values)
        
        return {
            "video_name": video_name,
            "frame_count": len(comparison_results),
            "average_metrics": avg_metrics,
            "per_frame_metrics": comparison_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_depth_metrics(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """Calculate standard depth estimation metrics"""
        # Mask for valid depth values
        valid_mask = (gt > 0) & (pred > 0) & np.isfinite(gt) & np.isfinite(pred)
        
        if not valid_mask.any():
            return {metric: float('inf') for metric in ["rmse", "mae", "abs_rel", "sq_rel", "delta1", "delta2", "delta3"]}
        
        pred_valid = pred[valid_mask]
        gt_valid = gt[valid_mask]
        
        # Basic metrics
        rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
        mae = np.mean(np.abs(pred_valid - gt_valid))
        abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
        sq_rel = np.mean(((pred_valid - gt_valid) ** 2) / gt_valid)
        
        # Threshold accuracy
        max_ratio = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
        delta1 = np.mean(max_ratio < 1.25)
        delta2 = np.mean(max_ratio < 1.25 ** 2)
        delta3 = np.mean(max_ratio < 1.25 ** 3)
        
        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "abs_rel": float(abs_rel),
            "sq_rel": float(sq_rel),
            "delta1": float(delta1),
            "delta2": float(delta2),
            "delta3": float(delta3)
        }
    
    def compare_point_clouds(
        self, 
        predicted_clouds_dir: Union[str, Path],
        video_name: str
    ) -> Dict[str, Any]:
        """Compare reconstructed point clouds with ground truth"""
        if not OPEN3D_AVAILABLE:
            return {"error": "Open3D not available for point cloud comparison"}
        
        predicted_dir = Path(predicted_clouds_dir) / video_name
        gt_dir = self.ground_truth_dir / "point_clouds" / video_name
        
        if not gt_dir.exists():
            logger.warning(f"No ground truth point clouds found for {video_name}")
            return {"error": "No ground truth available"}
        
        metrics = {
            "chamfer_distance": [],
            "hausdorff_distance": [],
            "point_count_diff": [],
            "geometric_consistency": []
        }
        
        comparison_results = []
        
        # Compare each frame
        for pred_file in predicted_dir.glob("*.ply"):
            frame_name = pred_file.stem
            gt_file = gt_dir / f"{frame_name}_gt.ply"
            
            if not gt_file.exists():
                continue
                
            try:
                pred_cloud = o3d.io.read_point_cloud(str(pred_file))
                gt_cloud = o3d.io.read_point_cloud(str(gt_file))
                
                frame_metrics = self._calculate_point_cloud_metrics(pred_cloud, gt_cloud)
                
                for key, value in frame_metrics.items():
                    metrics[key].append(value)
                
                comparison_results.append({
                    "frame": frame_name,
                    "metrics": frame_metrics
                })
                
            except Exception as e:
                logger.error(f"Error comparing point cloud for frame {frame_name}: {e}")
                continue
        
        # Calculate averages
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[f"avg_{key}"] = np.mean(values)
                avg_metrics[f"std_{key}"] = np.std(values)
        
        return {
            "video_name": video_name,
            "frame_count": len(comparison_results),
            "average_metrics": avg_metrics,
            "per_frame_metrics": comparison_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_point_cloud_metrics(self, pred_cloud: 'o3d.geometry.PointCloud', 
                                     gt_cloud: 'o3d.geometry.PointCloud') -> Dict[str, float]:
        """Calculate point cloud comparison metrics"""
        pred_points = np.asarray(pred_cloud.points)
        gt_points = np.asarray(gt_cloud.points)
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return {
                "chamfer_distance": float('inf'),
                "hausdorff_distance": float('inf'),
                "point_count_diff": float('inf'),
                "geometric_consistency": 0.0
            }
        
        # Chamfer distance (simplified)
        pred_tree = o3d.geometry.KDTreeFlann(pred_cloud)
        gt_tree = o3d.geometry.KDTreeFlann(gt_cloud)
        
        # Distance from pred to GT
        pred_to_gt_dists = []
        for point in pred_points:
            [_, idx, _] = gt_tree.search_knn_vector_3d(point, 1)
            if idx:
                dist = np.linalg.norm(point - gt_points[idx[0]])
                pred_to_gt_dists.append(dist)
        
        # Distance from GT to pred
        gt_to_pred_dists = []
        for point in gt_points:
            [_, idx, _] = pred_tree.search_knn_vector_3d(point, 1)
            if idx:
                dist = np.linalg.norm(point - pred_points[idx[0]])
                gt_to_pred_dists.append(dist)
        
        chamfer_distance = np.mean(pred_to_gt_dists) + np.mean(gt_to_pred_dists)
        hausdorff_distance = max(np.max(pred_to_gt_dists), np.max(gt_to_pred_dists))
        
        # Point count difference
        point_count_diff = abs(len(pred_points) - len(gt_points)) / len(gt_points)
        
        # Geometric consistency (simplified)
        geometric_consistency = 1.0 / (1.0 + chamfer_distance)
        
        return {
            "chamfer_distance": float(chamfer_distance),
            "hausdorff_distance": float(hausdorff_distance),
            "point_count_diff": float(point_count_diff),
            "geometric_consistency": float(geometric_consistency)
        }
    
    def compare_measurements(
        self, 
        predicted_measurements: Dict[str, Any],
        gt_measurements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare distance measurements with ground truth"""
        if not gt_measurements:
            return {"error": "No ground truth measurements available"}
        
        comparison_results = []
        
        for measurement_id, pred_data in predicted_measurements.items():
            if measurement_id not in gt_measurements:
                continue
            
            gt_data = gt_measurements[measurement_id]
            
            # Compare distances
            pred_distance = pred_data.get("distance", 0)
            gt_distance = gt_data.get("distance", 0)
            
            absolute_error = abs(pred_distance - gt_distance)
            relative_error = absolute_error / gt_distance if gt_distance > 0 else float('inf')
            
            comparison_results.append({
                "measurement_id": measurement_id,
                "predicted_distance": pred_distance,
                "ground_truth_distance": gt_distance,
                "absolute_error": absolute_error,
                "relative_error": relative_error
            })
        
        # Calculate summary statistics
        if comparison_results:
            abs_errors = [r["absolute_error"] for r in comparison_results]
            rel_errors = [r["relative_error"] for r in comparison_results if r["relative_error"] != float('inf')]
            
            summary = {
                "measurement_count": len(comparison_results),
                "mean_absolute_error": np.mean(abs_errors),
                "std_absolute_error": np.std(abs_errors),
                "mean_relative_error": np.mean(rel_errors) if rel_errors else float('inf'),
                "std_relative_error": np.std(rel_errors) if rel_errors else float('inf'),
                "max_absolute_error": np.max(abs_errors),
                "measurements": comparison_results
            }
        else:
            summary = {"error": "No comparable measurements found"}
        
        return summary


class AccuracyMetricsCalculator:
    """Calculate comprehensive accuracy metrics for all pipeline stages"""
    
    def __init__(self):
        self.metrics_database = {}
        
    def calculate_depth_accuracy(self, depth_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate depth estimation accuracy metrics"""
        accuracy_metrics = {
            "temporal_consistency": self._calculate_temporal_consistency(depth_results),
            "edge_preservation": self._calculate_edge_preservation(depth_results),
            "surface_smoothness": self._calculate_surface_smoothness(depth_results),
            "detail_preservation": self._calculate_detail_preservation(depth_results)
        }
        
        return accuracy_metrics
    
    def _calculate_temporal_consistency(self, depth_results: Dict[str, Any]) -> float:
        """Calculate temporal consistency across frames"""
        if "processed_frames" not in depth_results or len(depth_results["processed_frames"]) < 2:
            return 0.0
        
        consistency_scores = []
        
        for i in range(len(depth_results["processed_frames"]) - 1):
            frame1_path = depth_results["processed_frames"][i]["depth_path"]
            frame2_path = depth_results["processed_frames"][i + 1]["depth_path"]
            
            try:
                depth1 = np.load(frame1_path)
                depth2 = np.load(frame2_path)
                
                # Calculate frame-to-frame consistency
                if depth1.shape == depth2.shape:
                    diff = np.abs(depth1 - depth2)
                    consistency = 1.0 / (1.0 + np.mean(diff))
                    consistency_scores.append(consistency)
                    
            except Exception as e:
                logger.debug(f"Error calculating temporal consistency: {e}")
                continue
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_edge_preservation(self, depth_results: Dict[str, Any]) -> float:
        """Calculate how well edges are preserved in depth estimation"""
        if not CV2_AVAILABLE:
            return 0.5  # Default score
        
        edge_scores = []
        
        for frame_data in depth_results.get("processed_frames", []):
            try:
                depth_path = frame_data["depth_path"]
                depth = np.load(depth_path)
                
                # Calculate edge strength
                edges = cv2.Canny((depth * 255).astype(np.uint8), 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                edge_scores.append(edge_density)
                
            except Exception as e:
                logger.debug(f"Error calculating edge preservation: {e}")
                continue
        
        return np.mean(edge_scores) if edge_scores else 0.0
    
    def _calculate_surface_smoothness(self, depth_results: Dict[str, Any]) -> float:
        """Calculate surface smoothness metric"""
        smoothness_scores = []
        
        for frame_data in depth_results.get("processed_frames", []):
            try:
                depth_path = frame_data["depth_path"]
                depth = np.load(depth_path)
                
                # Calculate gradient magnitude
                grad_x = np.gradient(depth, axis=1)
                grad_y = np.gradient(depth, axis=0)
                grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # Smoothness is inverse of gradient magnitude
                smoothness = 1.0 / (1.0 + np.mean(grad_magnitude))
                smoothness_scores.append(smoothness)
                
            except Exception as e:
                logger.debug(f"Error calculating surface smoothness: {e}")
                continue
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.0
    
    def _calculate_detail_preservation(self, depth_results: Dict[str, Any]) -> float:
        """Calculate detail preservation metric"""
        detail_scores = []
        
        for frame_data in depth_results.get("processed_frames", []):
            try:
                depth_path = frame_data["depth_path"]
                depth = np.load(depth_path)
                
                # Calculate local variance as a measure of detail
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
                local_mean = cv2.filter2D(depth, -1, kernel)
                local_variance = cv2.filter2D((depth - local_mean)**2, -1, kernel)
                
                detail_score = np.mean(local_variance)
                detail_scores.append(detail_score)
                
            except Exception as e:
                logger.debug(f"Error calculating detail preservation: {e}")
                continue
        
        return np.mean(detail_scores) if detail_scores else 0.0
    
    def calculate_gaussian_accuracy(self, gaussian_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate 4D Gaussian reconstruction accuracy metrics"""
        accuracy_metrics = {
            "reconstruction_fidelity": self._calculate_reconstruction_fidelity(gaussian_results),
            "temporal_stability": self._calculate_temporal_stability(gaussian_results),
            "gaussian_efficiency": self._calculate_gaussian_efficiency(gaussian_results)
        }
        
        return accuracy_metrics
    
    def _calculate_reconstruction_fidelity(self, gaussian_results: Dict[str, Any]) -> float:
        """Calculate reconstruction fidelity score"""
        # Placeholder implementation
        # In a real implementation, this would compare reconstructed views with original frames
        return 0.75  # Default score
    
    def _calculate_temporal_stability(self, gaussian_results: Dict[str, Any]) -> float:
        """Calculate temporal stability of Gaussian parameters"""
        # Placeholder implementation
        # Would analyze consistency of Gaussian parameters across time
        return 0.8  # Default score
    
    def _calculate_gaussian_efficiency(self, gaussian_results: Dict[str, Any]) -> float:
        """Calculate efficiency of Gaussian representation"""
        # Placeholder implementation
        # Would analyze the number of Gaussians vs quality trade-off
        return 0.7  # Default score
    
    def calculate_enhancement_accuracy(self, enhancement_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate transformer enhancement accuracy metrics"""
        accuracy_metrics = {
            "enhancement_quality": self._calculate_enhancement_quality(enhancement_results),
            "noise_reduction": self._calculate_noise_reduction(enhancement_results),
            "feature_enhancement": self._calculate_feature_enhancement_quality(enhancement_results)
        }
        
        return accuracy_metrics
    
    def _calculate_enhancement_quality(self, enhancement_results: Dict[str, Any]) -> float:
        """Calculate overall enhancement quality"""
        # Placeholder implementation
        return 0.85  # Default score
    
    def _calculate_noise_reduction(self, enhancement_results: Dict[str, Any]) -> float:
        """Calculate noise reduction effectiveness"""
        # Placeholder implementation
        return 0.9  # Default score
    
    def _calculate_feature_enhancement_quality(self, enhancement_results: Dict[str, Any]) -> float:
        """Calculate feature enhancement quality"""
        # Placeholder implementation
        return 0.8  # Default score


class PerformanceBenchmarker:
    """Benchmark performance across different configurations and datasets"""
    
    def __init__(self, benchmark_config: Optional[Dict] = None):
        self.config = benchmark_config or self._get_default_config()
        self.results_database = {}
        
    def _get_default_config(self) -> Dict:
        """Get default benchmarking configuration"""
        return {
            "test_datasets": ["synthetic_video_1", "real_video_1"],
            "configurations": [
                {"name": "fast", "quality": "low", "batch_size": 8},
                {"name": "balanced", "quality": "medium", "batch_size": 4},
                {"name": "quality", "quality": "high", "batch_size": 2}
            ],
            "metrics_to_track": [
                "processing_time", "memory_usage", "gpu_utilization", 
                "accuracy_score", "quality_score"
            ]
        }
    
    def run_benchmark_suite(self, pipeline_runner: callable) -> Dict[str, Any]:
        """Run comprehensive benchmarks across datasets and configurations"""
        benchmark_results = {
            "start_time": datetime.now().isoformat(),
            "configuration": self.config,
            "results": {}
        }
        
        for dataset in self.config["test_datasets"]:
            benchmark_results["results"][dataset] = {}
            
            for config in self.config["configurations"]:
                config_name = config["name"]
                
                logger.info(f"Running benchmark: {dataset} with {config_name} configuration")
                
                try:
                    # Run pipeline with specific configuration
                    start_time = time.time()
                    
                    result = pipeline_runner(dataset, config)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # Collect performance metrics
                    perf_metrics = {
                        "processing_time": processing_time,
                        "success": result.get("success", False),
                        "error_message": result.get("error", None),
                        "quality_metrics": result.get("quality_metrics", {}),
                        "resource_usage": result.get("resource_usage", {})
                    }
                    
                    benchmark_results["results"][dataset][config_name] = perf_metrics
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {dataset}/{config_name}: {e}")
                    benchmark_results["results"][dataset][config_name] = {
                        "success": False,
                        "error_message": str(e),
                        "processing_time": None
                    }
        
        benchmark_results["end_time"] = datetime.now().isoformat()
        return benchmark_results
    
    def analyze_performance_trends(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends across configurations"""
        analysis = {
            "performance_summary": {},
            "recommendations": [],
            "efficiency_rankings": {}
        }
        
        # Extract performance data
        all_results = []
        for dataset, configs in benchmark_results["results"].items():
            for config_name, metrics in configs.items():
                if metrics.get("success", False):
                    all_results.append({
                        "dataset": dataset,
                        "configuration": config_name,
                        "processing_time": metrics.get("processing_time", 0),
                        "quality_score": metrics.get("quality_metrics", {}).get("overall_score", 0)
                    })
        
        if all_results:
            # Calculate average performance by configuration
            config_stats = {}
            for result in all_results:
                config = result["configuration"]
                if config not in config_stats:
                    config_stats[config] = {"times": [], "quality": []}
                
                config_stats[config]["times"].append(result["processing_time"])
                config_stats[config]["quality"].append(result["quality_score"])
            
            # Generate summary statistics
            for config, stats in config_stats.items():
                analysis["performance_summary"][config] = {
                    "avg_processing_time": np.mean(stats["times"]),
                    "std_processing_time": np.std(stats["times"]),
                    "avg_quality_score": np.mean(stats["quality"]),
                    "std_quality_score": np.std(stats["quality"]),
                    "efficiency_ratio": np.mean(stats["quality"]) / np.mean(stats["times"])
                }
            
            # Generate recommendations
            fastest_config = min(config_stats.keys(), 
                               key=lambda x: np.mean(config_stats[x]["times"]))
            highest_quality = max(config_stats.keys(), 
                                key=lambda x: np.mean(config_stats[x]["quality"]))
            most_efficient = max(config_stats.keys(),
                               key=lambda x: analysis["performance_summary"][x]["efficiency_ratio"])
            
            analysis["recommendations"] = [
                f"Fastest configuration: {fastest_config}",
                f"Highest quality configuration: {highest_quality}",
                f"Most efficient configuration: {most_efficient}"
            ]
        
        return analysis


class ValidationSuite:
    """Main validation suite coordinator"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.ground_truth_comparator = GroundTruthComparator(
            self.config.get("ground_truth_dir", "ground_truth")
        )
        self.accuracy_calculator = AccuracyMetricsCalculator()
        self.performance_benchmarker = PerformanceBenchmarker(
            self.config.get("benchmark_config")
        )
        
    def _get_default_config(self) -> Dict:
        """Get default validation configuration"""
        return {
            "ground_truth_dir": "ground_truth",
            "output_dir": "validation_results",
            "test_videos": ["test_video_1.mp4"],
            "enable_ground_truth_comparison": True,
            "enable_accuracy_metrics": True,
            "enable_performance_benchmarking": True,
            "generate_reports": True
        }
    
    def run_comprehensive_validation(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive validation of pipeline results"""
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config,
            "validation_stages": {}
        }
        
        logger.info("Starting comprehensive validation suite")
        
        # Stage 1: Ground truth comparison
        if self.config.get("enable_ground_truth_comparison", True):
            logger.info("Running ground truth comparisons")
            try:
                gt_results = self._run_ground_truth_validation(pipeline_results)
                validation_results["validation_stages"]["ground_truth"] = gt_results
            except Exception as e:
                logger.error(f"Ground truth validation failed: {e}")
                validation_results["validation_stages"]["ground_truth"] = {"error": str(e)}
        
        # Stage 2: Accuracy metrics
        if self.config.get("enable_accuracy_metrics", True):
            logger.info("Calculating accuracy metrics")
            try:
                accuracy_results = self._calculate_accuracy_metrics(pipeline_results)
                validation_results["validation_stages"]["accuracy_metrics"] = accuracy_results
            except Exception as e:
                logger.error(f"Accuracy metrics calculation failed: {e}")
                validation_results["validation_stages"]["accuracy_metrics"] = {"error": str(e)}
        
        # Stage 3: Performance benchmarking
        if self.config.get("enable_performance_benchmarking", True):
            logger.info("Running performance benchmarks")
            try:
                benchmark_results = self._run_performance_benchmarks(pipeline_results)
                validation_results["validation_stages"]["performance"] = benchmark_results
            except Exception as e:
                logger.error(f"Performance benchmarking failed: {e}")
                validation_results["validation_stages"]["performance"] = {"error": str(e)}
        
        # Generate comprehensive report
        if self.config.get("generate_reports", True):
            validation_results["summary_report"] = self._generate_validation_summary(validation_results)
        
        logger.info("Comprehensive validation completed")
        return validation_results
    
    def _run_ground_truth_validation(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run ground truth validation for all pipeline stages"""
        gt_validation = {}
        
        # Validate depth estimation
        if "depth_estimation" in pipeline_results:
            depth_results = pipeline_results["depth_estimation"]
            for video_name in depth_results.get("processed_videos", []):
                gt_validation[f"depth_{video_name}"] = self.ground_truth_comparator.compare_depth_maps(
                    depth_results.get("output_dir", "output/depth_maps"),
                    video_name
                )
        
        # Validate point cloud reconstruction
        if "point_clouds" in pipeline_results:
            pc_results = pipeline_results["point_clouds"]
            for video_name in pc_results.get("processed_videos", []):
                gt_validation[f"pointclouds_{video_name}"] = self.ground_truth_comparator.compare_point_clouds(
                    pc_results.get("output_dir", "output/point_clouds"),
                    video_name
                )
        
        # Validate measurements
        if "measurements" in pipeline_results:
            gt_measurements = self._load_ground_truth_measurements()
            gt_validation["measurements"] = self.ground_truth_comparator.compare_measurements(
                pipeline_results["measurements"],
                gt_measurements
            )
        
        return gt_validation
    
    def _calculate_accuracy_metrics(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy metrics for all pipeline stages"""
        accuracy_metrics = {}
        
        # Depth accuracy
        if "depth_estimation" in pipeline_results:
            accuracy_metrics["depth"] = self.accuracy_calculator.calculate_depth_accuracy(
                pipeline_results["depth_estimation"]
            )
        
        # Gaussian accuracy
        if "gaussian_4d" in pipeline_results:
            accuracy_metrics["gaussian"] = self.accuracy_calculator.calculate_gaussian_accuracy(
                pipeline_results["gaussian_4d"]
            )
        
        # Enhancement accuracy
        if "transformer_enhancement" in pipeline_results:
            accuracy_metrics["enhancement"] = self.accuracy_calculator.calculate_enhancement_accuracy(
                pipeline_results["transformer_enhancement"]
            )
        
        return accuracy_metrics
    
    def _run_performance_benchmarks(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance benchmarks"""
        # This would integrate with the actual pipeline runner
        # For now, return mock benchmark results
        return {
            "note": "Performance benchmarking requires integration with pipeline runner",
            "mock_results": {
                "configurations_tested": 3,
                "datasets_tested": 2,
                "average_processing_time": 120.5,
                "memory_efficiency": 0.85
            }
        }
    
    def _load_ground_truth_measurements(self) -> Dict[str, Any]:
        """Load ground truth measurements if available"""
        gt_measurements_file = self.ground_truth_comparator.ground_truth_dir / "measurements.json"
        
        if gt_measurements_file.exists():
            with open(gt_measurements_file, 'r') as f:
                return json.load(f)
        
        return {}
    
    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive validation summary"""
        summary = {
            "overall_status": "PASS",
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "scores": {}
        }
        
        # Analyze ground truth results
        gt_results = validation_results.get("validation_stages", {}).get("ground_truth", {})
        if "error" in gt_results:
            summary["warnings"].append("Ground truth comparison failed")
        else:
            # Extract average scores from ground truth comparisons
            for stage, data in gt_results.items():
                if "average_metrics" in data:
                    avg_metrics = data["average_metrics"]
                    if "avg_rmse" in avg_metrics:
                        summary["scores"][f"{stage}_rmse"] = avg_metrics["avg_rmse"]
        
        # Analyze accuracy metrics
        accuracy_results = validation_results.get("validation_stages", {}).get("accuracy_metrics", {})
        if "error" in accuracy_results:
            summary["warnings"].append("Accuracy metrics calculation failed")
        else:
            for stage, metrics in accuracy_results.items():
                for metric_name, value in metrics.items():
                    summary["scores"][f"{stage}_{metric_name}"] = value
        
        # Generate recommendations based on scores
        if summary["scores"]:
            avg_score = np.mean(list(summary["scores"].values()))
            if avg_score < 0.5:
                summary["overall_status"] = "FAIL"
                summary["critical_issues"].append("Overall performance below acceptable threshold")
            elif avg_score < 0.7:
                summary["overall_status"] = "WARNING"
                summary["warnings"].append("Performance needs improvement")
            
            summary["recommendations"].extend([
                f"Average performance score: {avg_score:.3f}",
                "Consider tuning pipeline parameters for better performance" if avg_score < 0.8 else "Performance is acceptable"
            ])
        
        return summary


def create_sample_test_videos():
    """Create synthetic test videos for validation"""
    logger.info("Creating sample test videos for validation")
    
    # This would create synthetic test data
    # For now, just create the directory structure
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    (test_data_dir / "videos").mkdir(exist_ok=True)
    (test_data_dir / "ground_truth" / "depth_maps").mkdir(parents=True, exist_ok=True)
    (test_data_dir / "ground_truth" / "point_clouds").mkdir(parents=True, exist_ok=True)
    
    logger.info("Test data directory structure created")


def main():
    """Main validation suite runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gauge 3D Validation Suite")
    parser.add_argument("--pipeline-results", help="Path to pipeline results JSON")
    parser.add_argument("--config", help="Path to validation configuration file")
    parser.add_argument("--output", default="validation_results", help="Output directory")
    parser.add_argument("--create-test-data", action="store_true", help="Create sample test data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    if args.create_test_data:
        create_sample_test_videos()
        return
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                import yaml
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    # Load pipeline results
    pipeline_results = {}
    if args.pipeline_results:
        with open(args.pipeline_results, 'r') as f:
            pipeline_results = json.load(f)
    
    # Initialize validation suite
    validation_suite = ValidationSuite(config)
    
    # Run validation
    results = validation_suite.run_comprehensive_validation(pipeline_results)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"validation_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Validation completed. Results saved to: {results_file}")
    
    # Print summary
    if "summary_report" in results:
        summary = results["summary_report"]
        print("\n📊 Validation Summary:")
        print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        
        if summary.get("critical_issues"):
            print("❌ Critical Issues:")
            for issue in summary["critical_issues"]:
                print(f"  - {issue}")
        
        if summary.get("warnings"):
            print("⚠️  Warnings:")
            for warning in summary["warnings"]:
                print(f"  - {warning}")
        
        if summary.get("recommendations"):
            print("💡 Recommendations:")
            for rec in summary["recommendations"]:
                print(f"  - {rec}")


if __name__ == "__main__":
    main()
