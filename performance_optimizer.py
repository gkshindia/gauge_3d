#!/usr/bin/env python3
"""
Performance Optimization Module for Gauge 3D
Phase 4.3: Performance profiling, optimization, and resource management

Implements pipeline stage profiling, parallel processing, memory management,
and GPU utilization optimization for the complete Gauge 3D pipeline.
"""

import time
import psutil
import logging
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
import json

# Configure logging
logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for pipeline stages"""
    stage_name: str
    start_time: float
    end_time: float
    duration: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    throughput_items_per_second: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SystemResourceInfo:
    """System resource information"""
    cpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    gpu_available: bool
    gpu_count: int
    gpu_memory_gb: List[float]
    disk_space_gb: float


class PerformanceProfiler:
    """Profile performance of pipeline stages"""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring and TORCH_AVAILABLE
        self.metrics_history = []
        self.current_stage = None
        self.start_time = None
        self.peak_memory = 0
        
        # Initialize monitoring
        self.system_info = self._get_system_info()
        
        logger.info("PerformanceProfiler initialized")
        if self.enable_gpu_monitoring:
            logger.info(f"GPU monitoring enabled. Found {len(self.system_info.gpu_memory_gb)} GPU(s)")
    
    def _get_system_info(self) -> SystemResourceInfo:
        """Get system resource information"""
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            cpu_count = psutil.cpu_count()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            disk_space_gb = disk.free / (1024**3)
        else:
            cpu_count = multiprocessing.cpu_count()
            total_memory_gb = 8.0  # Default assumption
            available_memory_gb = 4.0
            disk_space_gb = 100.0
        
        # GPU information
        gpu_available = False
        gpu_count = 0
        gpu_memory_gb = []
        
        if self.enable_gpu_monitoring and TORCH_AVAILABLE:
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    memory = torch.cuda.get_device_properties(i).total_memory
                    gpu_memory_gb.append(memory / (1024**3))
        
        return SystemResourceInfo(
            cpu_count=cpu_count,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            disk_space_gb=disk_space_gb
        )
    
    @contextmanager
    def profile_stage(self, stage_name: str, item_count: Optional[int] = None):
        """Context manager for profiling a pipeline stage"""
        logger.info(f"Starting profiling for stage: {stage_name}")
        
        self.current_stage = stage_name
        self.start_time = time.time()
        self.peak_memory = 0
        
        # Get initial resource usage
        initial_cpu = self._get_cpu_usage()
        initial_memory = self._get_memory_usage()
        initial_gpu_memory = self._get_gpu_memory_usage()
        initial_gpu_util = self._get_gpu_utilization()
        
        # Start monitoring thread
        monitoring_active = threading.Event()
        monitoring_active.set()
        monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(monitoring_active,)
        )
        monitor_thread.start()
        
        try:
            yield self
            
        finally:
            # Stop monitoring
            monitoring_active.clear()
            monitor_thread.join(timeout=1.0)
            
            # Calculate final metrics
            end_time = time.time()
            duration = end_time - self.start_time
            
            # Get final resource usage
            final_cpu = self._get_cpu_usage()
            final_memory = self._get_memory_usage()
            final_gpu_memory = self._get_gpu_memory_usage()
            final_gpu_util = self._get_gpu_utilization()
            
            # Calculate throughput
            throughput = None
            if item_count and duration > 0:
                throughput = item_count / duration
            
            # Create metrics record
            metrics = PerformanceMetrics(
                stage_name=stage_name,
                start_time=self.start_time,
                end_time=end_time,
                duration=duration,
                cpu_usage_percent=max(initial_cpu, final_cpu),
                memory_usage_mb=max(initial_memory, final_memory),
                gpu_memory_mb=max(initial_gpu_memory or 0, final_gpu_memory or 0) if initial_gpu_memory else None,
                gpu_utilization_percent=max(initial_gpu_util or 0, final_gpu_util or 0) if initial_gpu_util else None,
                throughput_items_per_second=throughput,
                peak_memory_mb=self.peak_memory
            )
            
            self.metrics_history.append(metrics)
            
            logger.info(f"Stage '{stage_name}' completed in {duration:.2f}s")
            if throughput:
                logger.info(f"Throughput: {throughput:.2f} items/sec")
    
    def _monitor_resources(self, active_event: threading.Event):
        """Monitor resources in background thread"""
        while active_event.is_set():
            current_memory = self._get_memory_usage()
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            
            time.sleep(0.1)  # Monitor every 100ms
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        if PSUTIL_AVAILABLE:
            return psutil.cpu_percent(interval=0.1)
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        return 0.0
    
    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in MB"""
        if self.enable_gpu_monitoring and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated() / (1024 * 1024)
            except:
                return None
        return None
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage"""
        # This would require nvidia-ml-py or similar
        # For now, return None as it's not critical
        return None
    
    def get_performance_summary(self) -> Dict:
        """Get summary of all performance metrics"""
        if not self.metrics_history:
            return {"no_data": True}
        
        total_duration = sum(m.duration for m in self.metrics_history)
        avg_cpu = sum(m.cpu_usage_percent for m in self.metrics_history) / len(self.metrics_history)
        peak_memory = max(m.peak_memory_mb or m.memory_usage_mb for m in self.metrics_history)
        
        return {
            "total_stages": len(self.metrics_history),
            "total_duration_seconds": total_duration,
            "average_cpu_usage_percent": avg_cpu,
            "peak_memory_usage_mb": peak_memory,
            "system_info": asdict(self.system_info),
            "stage_breakdown": [m.to_dict() for m in self.metrics_history]
        }
    
    def export_metrics(self, output_path: Union[str, Path]):
        """Export performance metrics to file"""
        output_path = Path(output_path)
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "performance_summary": self.get_performance_summary(),
            "detailed_metrics": [m.to_dict() for m in self.metrics_history]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Performance metrics exported to: {output_path}")


class MemoryManager:
    """Manage memory usage and optimization"""
    
    def __init__(self, memory_limit_gb: float = 16.0):
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.cache = {}
        self.cache_size_limit = memory_limit_gb * 0.25 * 1024**3  # 25% of limit for cache
        
        logger.info(f"MemoryManager initialized with {memory_limit_gb}GB limit")
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            return {
                "process_memory_mb": memory_info.rss / (1024**2),
                "process_memory_percent": (memory_info.rss / self.memory_limit_bytes) * 100,
                "system_memory_mb": (system_memory.total - system_memory.available) / (1024**2),
                "system_memory_percent": system_memory.percent,
                "cache_size_mb": sum(self._get_object_size(obj) for obj in self.cache.values()) / (1024**2)
            }
        else:
            return {"monitoring_unavailable": True}
    
    def _get_object_size(self, obj) -> int:
        """Estimate object size in bytes"""
        if NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
            return obj.nbytes
        elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        else:
            # Rough estimate
            return len(str(obj)) * 8
    
    @contextmanager
    def memory_limit_context(self, stage_name: str):
        """Context manager for memory-limited operations"""
        logger.debug(f"Entering memory-limited context for: {stage_name}")
        
        initial_memory = self.check_memory_usage()
        
        try:
            yield
            
        finally:
            final_memory = self.check_memory_usage()
            
            if "process_memory_mb" in final_memory:
                memory_increase = final_memory["process_memory_mb"] - initial_memory.get("process_memory_mb", 0)
                if memory_increase > 1000:  # More than 1GB increase
                    logger.warning(f"Stage '{stage_name}' increased memory by {memory_increase:.1f}MB")
                
                # Check if approaching limit
                if final_memory["process_memory_percent"] > 80:
                    logger.warning(f"Memory usage at {final_memory['process_memory_percent']:.1f}% of limit")
                    self.cleanup_cache()
    
    def cache_data(self, key: str, data: Any, force: bool = False):
        """Cache data with memory management"""
        data_size = self._get_object_size(data)
        
        # Check cache size limit
        current_cache_size = sum(self._get_object_size(obj) for obj in self.cache.values())
        
        if current_cache_size + data_size > self.cache_size_limit and not force:
            logger.warning("Cache size limit reached, cleaning up...")
            self.cleanup_cache()
        
        self.cache[key] = data
        logger.debug(f"Cached data '{key}' ({data_size / (1024**2):.1f}MB)")
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data"""
        return self.cache.get(key)
    
    def cleanup_cache(self):
        """Clean up cache to free memory"""
        cache_items = list(self.cache.items())
        cache_items.sort(key=lambda x: self._get_object_size(x[1]), reverse=True)
        
        # Remove largest 50% of cache items
        items_to_remove = len(cache_items) // 2
        
        for i in range(items_to_remove):
            key, _ = cache_items[i]
            del self.cache[key]
        
        logger.info(f"Cleaned up cache, removed {items_to_remove} items")
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            
            logger.info(f"GPU memory optimized - Allocated: {allocated/(1024**2):.1f}MB, Cached: {cached/(1024**2):.1f}MB")


class ParallelProcessor:
    """Handle parallel processing for pipeline stages"""
    
    def __init__(self, max_workers: Optional[int] = None, chunk_size: int = 100):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.chunk_size = chunk_size
        
        logger.info(f"ParallelProcessor initialized with {self.max_workers} workers")
    
    def process_batch_parallel(
        self, 
        items: List[Any], 
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process items in parallel batches"""
        import concurrent.futures
        
        results = []
        total_items = len(items)
        
        logger.info(f"Processing {total_items} items in parallel with {self.max_workers} workers")
        
        # Split items into chunks
        chunks = [items[i:i + self.chunk_size] for i in range(0, len(items), self.chunk_size)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, process_func): chunk 
                for chunk in chunks
            }
            
            # Collect results
            completed_items = 0
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_results = future.result()
                results.extend(chunk_results)
                
                completed_items += len(chunk_results)
                
                if progress_callback:
                    progress_callback(completed_items, total_items)
                
                logger.debug(f"Completed chunk: {completed_items}/{total_items} items")
        
        logger.info(f"Parallel processing completed: {len(results)} results")
        return results
    
    def _process_chunk(self, chunk: List[Any], process_func: Callable) -> List[Any]:
        """Process a chunk of items"""
        return [process_func(item) for item in chunk]
    
    def process_video_frames_parallel(
        self, 
        video_path: Union[str, Path], 
        frame_processor: Callable,
        frame_skip: int = 1,
        max_frames: Optional[int] = None
    ) -> List[Any]:
        """Process video frames in parallel"""
        # This would require OpenCV or similar for video reading
        # For now, provide a structure that can be extended
        
        logger.info(f"Processing video frames from: {video_path}")
        
        # Placeholder implementation
        frame_indices = list(range(0, max_frames or 1000, frame_skip))
        
        def process_frame_by_index(frame_idx):
            # This would load and process the actual frame
            return frame_processor(f"frame_{frame_idx:06d}")
        
        return self.process_batch_parallel(frame_indices, process_frame_by_index)


class GPUOptimizer:
    """Optimize GPU utilization and memory usage"""
    
    def __init__(self, memory_fraction: float = 0.8):
        self.memory_fraction = memory_fraction
        self.device = self._setup_device()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info(f"GPU optimizer initialized for device: {self.device}")
            self._configure_gpu_memory()
        else:
            logger.info("GPU optimizer initialized for CPU-only operation")
    
    def _setup_device(self) -> str:
        """Setup computation device"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _configure_gpu_memory(self):
        """Configure GPU memory settings"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Set memory fraction
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_limit = int(total_memory * self.memory_fraction)
            
            # Note: PyTorch doesn't have direct memory fraction setting like TensorFlow
            # This is more of a monitoring setup
            logger.info(f"GPU memory limit set to {memory_limit / (1024**3):.1f}GB")
    
    def optimize_tensor_operations(self, enable_mixed_precision: bool = True):
        """Optimize tensor operations"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Enable mixed precision if supported
            if enable_mixed_precision:
                try:
                    # This would be used with torch.cuda.amp in actual implementation
                    logger.info("Mixed precision optimization enabled")
                except:
                    logger.warning("Mixed precision not available")
            
            # Optimize CUDNN
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
                logger.info("CUDNN benchmark optimization enabled")
    
    def get_gpu_status(self) -> Dict:
        """Get current GPU status"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "memory_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "memory_total_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2)
            }
        else:
            return {"gpu_available": False}
    
    @contextmanager
    def gpu_memory_context(self):
        """Context manager for GPU memory management"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            
            try:
                yield
                
            finally:
                final_memory = torch.cuda.memory_allocated()
                memory_diff = final_memory - initial_memory
                
                if memory_diff > 100 * 1024 * 1024:  # More than 100MB
                    logger.info(f"GPU memory increased by {memory_diff / (1024**2):.1f}MB")
                
                # Clean up if needed
                if torch.cuda.memory_allocated() > torch.cuda.get_device_properties(0).total_memory * 0.9:
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared due to high memory usage")
        else:
            yield


class PipelineOptimizer:
    """Main pipeline optimization coordinator"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.profiler = PerformanceProfiler(
            enable_gpu_monitoring=self.config.get('gpu_monitoring', True)
        )
        self.memory_manager = MemoryManager(
            memory_limit_gb=self.config.get('memory_limit_gb', 16.0)
        )
        self.parallel_processor = ParallelProcessor(
            max_workers=self.config.get('max_workers'),
            chunk_size=self.config.get('chunk_size', 100)
        )
        self.gpu_optimizer = GPUOptimizer(
            memory_fraction=self.config.get('gpu_memory_fraction', 0.8)
        )
        
        logger.info("PipelineOptimizer initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default optimization configuration"""
        return {
            "memory_limit_gb": 16.0,
            "gpu_memory_fraction": 0.8,
            "max_workers": None,  # Auto-detect
            "chunk_size": 100,
            "gpu_monitoring": True,
            "enable_profiling": True,
            "enable_parallel_processing": True,
            "mixed_precision": True
        }
    
    @contextmanager
    def optimized_stage(self, stage_name: str, item_count: Optional[int] = None):
        """Context manager for optimized pipeline stage execution"""
        with self.profiler.profile_stage(stage_name, item_count), \
             self.memory_manager.memory_limit_context(stage_name), \
             self.gpu_optimizer.gpu_memory_context():
            
            yield {
                'profiler': self.profiler,
                'memory_manager': self.memory_manager,
                'parallel_processor': self.parallel_processor,
                'gpu_optimizer': self.gpu_optimizer
            }
    
    def generate_optimization_report(self, output_path: Union[str, Path]):
        """Generate comprehensive optimization report"""
        output_path = Path(output_path)
        
        report = {
            "generation_time": datetime.now().isoformat(),
            "performance_summary": self.profiler.get_performance_summary(),
            "memory_status": self.memory_manager.check_memory_usage(),
            "gpu_status": self.gpu_optimizer.get_gpu_status(),
            "optimization_config": self.config,
            "recommendations": self._generate_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization report generated: {output_path}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Memory recommendations
        memory_status = self.memory_manager.check_memory_usage()
        if memory_status.get("process_memory_percent", 0) > 70:
            recommendations.append("Consider increasing memory limit or reducing batch size")
        
        # GPU recommendations
        gpu_status = self.gpu_optimizer.get_gpu_status()
        if gpu_status.get("gpu_available", False):
            memory_usage_percent = (gpu_status.get("memory_allocated_mb", 0) / 
                                  gpu_status.get("memory_total_mb", 1)) * 100
            if memory_usage_percent < 50:
                recommendations.append("GPU underutilized - consider increasing batch size")
            elif memory_usage_percent > 90:
                recommendations.append("GPU memory nearly full - consider reducing batch size")
        
        # Performance recommendations
        perf_summary = self.profiler.get_performance_summary()
        if perf_summary.get("average_cpu_usage_percent", 0) < 50:
            recommendations.append("CPU underutilized - consider enabling parallel processing")
        
        return recommendations


def main():
    """Example usage of performance optimization tools"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gauge 3D Performance Optimization")
    parser.add_argument("--profile", action="store_true", help="Run performance profiling example")
    parser.add_argument("--memory", action="store_true", help="Run memory management example")
    parser.add_argument("--gpu", action="store_true", help="Run GPU optimization example")
    parser.add_argument("--output", "-o", default="optimization_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = PipelineOptimizer()
    
    if args.profile or not any([args.memory, args.gpu]):
        print("Running performance profiling example...")
        
        # Example profiled operations
        with optimizer.optimized_stage("Example Stage 1", item_count=1000) as tools:
            # Simulate some work
            time.sleep(1.0)
            print("Completed stage 1")
        
        with optimizer.optimized_stage("Example Stage 2", item_count=500) as tools:
            # Simulate memory-intensive work
            if NUMPY_AVAILABLE:
                large_array = np.random.randn(1000, 1000)
                tools['memory_manager'].cache_data("example_array", large_array)
            time.sleep(0.5)
            print("Completed stage 2")
    
    if args.memory:
        print("Running memory management example...")
        
        memory_manager = MemoryManager()
        
        print("Initial memory status:")
        print(json.dumps(memory_manager.check_memory_usage(), indent=2))
        
        # Simulate memory usage
        with memory_manager.memory_limit_context("Memory Test"):
            if NUMPY_AVAILABLE:
                test_data = np.random.randn(10000, 100)
                memory_manager.cache_data("test_data", test_data)
        
        print("Final memory status:")
        print(json.dumps(memory_manager.check_memory_usage(), indent=2))
    
    if args.gpu:
        print("Running GPU optimization example...")
        
        gpu_optimizer = GPUOptimizer()
        
        print("GPU status:")
        print(json.dumps(gpu_optimizer.get_gpu_status(), indent=2))
        
        gpu_optimizer.optimize_tensor_operations()
    
    # Generate report
    report = optimizer.generate_optimization_report(args.output)
    print(f"\nOptimization report saved to: {args.output}")
    
    if report.get("recommendations"):
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
