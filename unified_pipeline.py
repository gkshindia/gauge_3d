#!/usr/bin/env python3
"""
Gauge 3D Unified Pipeline

End-to-end pipeline connecting all three phases:
Phase 1: Depth Estimation
Phase 2: 4D Gaussian Generation  
Phase 3: Transformer Enhancement
Phase 4: Integration and Optimization
"""

import sys
import json
import yaml
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import time

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass


class ProgressTracker:
    """Track progress throughout the pipeline"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []
        
    def update(self, step_name: str, status: str = "complete"):
        """Update progress with step information"""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        self.step_times.append(elapsed)
        
        progress = (self.current_step / self.total_steps) * 100
        
        logger.info(f"Progress: {progress:.1f}% - Step {self.current_step}/{self.total_steps}: {step_name} ({status})")
        logger.info(f"Elapsed time: {elapsed:.2f}s, Step time: {self.step_times[-1] if len(self.step_times) > 1 else elapsed:.2f}s")
        
        return {
            "step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": progress,
            "step_name": step_name,
            "status": status,
            "elapsed_time": elapsed,
            "estimated_remaining": ((elapsed / self.current_step) * (self.total_steps - self.current_step)) if self.current_step > 0 else 0
        }


class ConfigurationManager:
    """Manage configurations across all pipeline phases"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else Path("config/pipeline_config.yaml")
        self.config = self._load_configuration()
        
    def _load_configuration(self) -> Dict:
        """Load master configuration file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration"""
        return {
            "pipeline": {
                "input_video_path": "vinput/",
                "output_base_path": "output/",
                "temp_path": "temp/",
                "enable_gpu": True,
                "batch_size": 4,
                "max_frames": None,
                "resume_from_checkpoint": True
            },
            "depth_estimation": {
                "config_file": "depth_estimation/config/dav_config.yaml",
                "model": "depth_anything_v2",
                "frame_skip": 6,
                "output_format": "npy"
            },
            "gaussian_4d": {
                "config_file": "4d_gaussian/config/gaussian_config.py",
                "max_gaussians": 100000,
                "optimization_iterations": 1000,
                "temporal_consistency": True
            },
            "transformer": {
                "config_file": "transformer/config/transformer_config.yaml",
                "enhancement_enabled": True,
                "p4transformer_model": "placeholder",
                "quality_threshold": 0.5
            },
            "validation": {
                "enable_quality_checks": True,
                "enable_progress_saving": True,
                "validate_outputs": True
            },
            "performance": {
                "enable_profiling": False,
                "memory_limit_gb": 16,
                "parallel_processing": True,
                "gpu_memory_fraction": 0.8
            }
        }
    
    def get_phase_config(self, phase: str) -> Dict:
        """Get configuration for specific phase"""
        return self.config.get(phase, {})
    
    def save_configuration(self):
        """Save current configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


class PipelineValidator:
    """Validate pipeline inputs, outputs, and intermediate results"""
    
    @staticmethod
    def validate_input_video(video_path: Path) -> bool:
        """Validate input video file"""
        if not video_path.exists():
            raise PipelineError(f"Input video not found: {video_path}")
        
        if video_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
            raise PipelineError(f"Unsupported video format: {video_path.suffix}")
        
        # Check file size
        size_mb = video_path.stat().st_size / (1024 * 1024)
        if size_mb > 10000:  # 10GB limit
            logger.warning(f"Large video file detected: {size_mb:.1f}MB")
        
        return True
    
    @staticmethod
    def validate_depth_output(depth_output_dir: Path) -> bool:
        """Validate depth estimation output"""
        if not depth_output_dir.exists():
            raise PipelineError(f"Depth output directory not found: {depth_output_dir}")
        
        depth_files = list(depth_output_dir.glob("*.npy"))
        if len(depth_files) == 0:
            raise PipelineError("No depth map files found")
        
        # Check a sample depth file
        sample_depth = np.load(depth_files[0])
        if sample_depth.ndim != 2:
            raise PipelineError(f"Invalid depth map dimensions: {sample_depth.shape}")
        
        logger.info(f"Validated {len(depth_files)} depth maps")
        return True
    
    @staticmethod
    def validate_gaussian_output(gaussian_output_dir: Path) -> bool:
        """Validate 4D Gaussian output"""
        if not gaussian_output_dir.exists():
            raise PipelineError(f"Gaussian output directory not found: {gaussian_output_dir}")
        
        # Check for required files
        required_files = ["positions.npy", "colors.npy"]
        for file_name in required_files:
            file_path = gaussian_output_dir / file_name
            if not file_path.exists():
                raise PipelineError(f"Required Gaussian file not found: {file_name}")
        
        logger.info("Validated Gaussian output files")
        return True
    
    @staticmethod
    def validate_transformer_output(transformer_output_dir: Path) -> bool:
        """Validate transformer enhancement output"""
        if not transformer_output_dir.exists():
            logger.warning(f"Transformer output directory not found: {transformer_output_dir}")
            return False
        
        output_files = list(transformer_output_dir.glob("*.npy")) + list(transformer_output_dir.glob("*.pth"))
        if len(output_files) == 0:
            logger.warning("No transformer output files found")
            return False
        
        logger.info(f"Validated {len(output_files)} transformer output files")
        return True


class UnifiedPipeline:
    """Unified pipeline orchestrating all three phases"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_manager = ConfigurationManager(config_path)
        self.config = self.config_manager.config
        self.validator = PipelineValidator()
        self.progress_tracker = None
        self.results = {}
        
        # Setup paths
        self.setup_paths()
        
        logger.info("Unified Pipeline initialized")
    
    def setup_paths(self):
        """Setup and validate required directories"""
        pipeline_config = self.config['pipeline']
        
        self.input_path = Path(pipeline_config['input_video_path'])
        self.output_path = Path(pipeline_config['output_base_path'])
        self.temp_path = Path(pipeline_config['temp_path'])
        
        # Create output directories
        for path in [self.output_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(
        self, 
        input_video: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Run the complete pipeline from video input to enhanced 3D reconstruction
        
        Args:
            input_video: Path to input video file
            output_dir: Optional custom output directory
            
        Returns:
            Dictionary containing results from all phases
        """
        start_time = time.time()
        input_video = Path(input_video)
        
        if output_dir:
            self.output_path = Path(output_dir)
            self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracking (8 main steps)
        self.progress_tracker = ProgressTracker(8)
        
        logger.info(f"Starting full pipeline for video: {input_video}")
        logger.info(f"Output directory: {self.output_path}")
        
        try:
            # Step 1: Validate input
            progress = self.progress_tracker.update("Input Validation")
            self.validator.validate_input_video(input_video)
            
            # Step 2: Phase 1 - Depth Estimation
            progress = self.progress_tracker.update("Depth Estimation - Setup")
            depth_results = self.run_depth_estimation(input_video)
            self.results['depth_estimation'] = depth_results
            
            # Step 3: Validate depth output
            progress = self.progress_tracker.update("Depth Validation")
            self.validator.validate_depth_output(depth_results['output_dir'])
            
            # Step 4: Phase 2 - 4D Gaussian Generation
            progress = self.progress_tracker.update("4D Gaussian Generation")
            gaussian_results = self.run_gaussian_generation(depth_results)
            self.results['gaussian_4d'] = gaussian_results
            
            # Step 5: Validate Gaussian output
            progress = self.progress_tracker.update("Gaussian Validation")
            self.validator.validate_gaussian_output(gaussian_results['output_dir'])
            
            # Step 6: Phase 3 - Transformer Enhancement
            progress = self.progress_tracker.update("Transformer Enhancement")
            transformer_results = self.run_transformer_enhancement(gaussian_results)
            self.results['transformer'] = transformer_results
            
            # Step 7: Validate transformer output
            progress = self.progress_tracker.update("Enhancement Validation")
            self.validator.validate_transformer_output(transformer_results['output_dir'])
            
            # Step 8: Generate final results
            progress = self.progress_tracker.update("Results Generation")
            final_results = self.generate_final_results()
            
            total_time = time.time() - start_time
            
            # Compile complete results
            complete_results = {
                'success': True,
                'input_video': str(input_video),
                'output_directory': str(self.output_path),
                'processing_time': total_time,
                'phases': self.results,
                'final_outputs': final_results,
                'pipeline_config': self.config,
                'progress_log': progress
            }
            
            # Save results
            self.save_pipeline_results(complete_results)
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            return complete_results
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            error_results = {
                'success': False,
                'error': str(e),
                'error_traceback': traceback.format_exc(),
                'partial_results': self.results,
                'processing_time': time.time() - start_time
            }
            
            self.save_pipeline_results(error_results)
            raise PipelineError(error_msg) from e
    
    def run_depth_estimation(self, input_video: Path) -> Dict:
        """Run Phase 1: Depth Estimation"""
        logger.info("Starting depth estimation phase...")
        
        try:
            from depth_estimation.main_pipeline import DepthEstimationPipeline
            
            depth_config = self.config['depth_estimation']
            depth_pipeline = DepthEstimationPipeline()
            
            output_dir = self.output_path / "depth_maps" / input_video.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run depth estimation
            results = depth_pipeline.process_video(
                video_path=str(input_video),
                output_dir=str(output_dir),
                frame_skip=depth_config.get('frame_skip', 6)
            )
            
            return {
                'output_dir': output_dir,
                'num_frames': results.get('num_frames', 0),
                'processing_time': results.get('processing_time', 0),
                'config': depth_config
            }
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise PipelineError(f"Depth estimation failed: {e}") from e
    
    def run_gaussian_generation(self, depth_results: Dict) -> Dict:
        """Run Phase 2: 4D Gaussian Generation"""
        logger.info("Starting 4D Gaussian generation phase...")
        
        try:
            from gaussian_4d.run_4d_gaussian import GaussianGenerator
            
            gaussian_config = self.config['gaussian_4d']
            generator = GaussianGenerator()
            
            output_dir = self.output_path / "gaussian_reconstruction" / "enhanced"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run Gaussian generation
            results = generator.generate_from_depth_maps(
                depth_dir=depth_results['output_dir'],
                output_dir=str(output_dir),
                max_gaussians=gaussian_config.get('max_gaussians', 100000)
            )
            
            return {
                'output_dir': output_dir,
                'num_gaussians': results.get('num_gaussians', 0),
                'processing_time': results.get('processing_time', 0),
                'config': gaussian_config
            }
            
        except Exception as e:
            logger.error(f"Gaussian generation failed: {e}")
            raise PipelineError(f"Gaussian generation failed: {e}") from e
    
    def run_transformer_enhancement(self, gaussian_results: Dict) -> Dict:
        """Run Phase 3: Transformer Enhancement"""
        logger.info("Starting transformer enhancement phase...")
        
        try:
            from transformer.point_cloud_extractor import PointCloudExtractor
            from transformer.enhancement_pipeline import EnhancementPipeline
            from transformer.reconstruction_pipeline import ReconstructionPipeline
            
            transformer_config = self.config['transformer']
            
            output_dir = self.output_path / "transformer_enhanced"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Extract point clouds
            extractor = PointCloudExtractor()
            extraction_result = extractor.extract_from_gaussians(
                str(gaussian_results['output_dir'])
            )
            
            # Step 2: Enhance point clouds
            enhancer = EnhancementPipeline()
            enhanced_clouds = enhancer.process(extraction_result['point_clouds'])
            
            # Step 3: Reconstruct enhanced Gaussians
            reconstructor = ReconstructionPipeline()
            enhanced_gaussians = reconstructor.reconstruct(enhanced_clouds, str(output_dir))
            
            return {
                'output_dir': output_dir,
                'num_enhanced_clouds': len(enhanced_clouds),
                'num_enhanced_gaussians': enhanced_gaussians.get('num_frames', 0),
                'processing_time': time.time(),
                'config': transformer_config
            }
            
        except Exception as e:
            logger.error(f"Transformer enhancement failed: {e}")
            # Don't fail the entire pipeline for transformer issues
            logger.warning("Continuing pipeline without enhancement")
            
            return {
                'output_dir': self.output_path / "transformer_enhanced",
                'num_enhanced_clouds': 0,
                'error': str(e),
                'config': transformer_config
            }
    
    def generate_final_results(self) -> Dict:
        """Generate final pipeline results and summaries"""
        logger.info("Generating final results...")
        
        final_outputs = {
            'depth_maps': self.output_path / "depth_maps",
            'gaussian_reconstruction': self.output_path / "gaussian_reconstruction", 
            'enhanced_reconstruction': self.output_path / "transformer_enhanced",
            'pipeline_log': Path("pipeline.log"),
            'results_summary': self.output_path / "pipeline_results.json"
        }
        
        # Generate summary statistics
        summary_stats = {
            'total_frames_processed': self.results.get('depth_estimation', {}).get('num_frames', 0),
            'total_gaussians_generated': self.results.get('gaussian_4d', {}).get('num_gaussians', 0),
            'enhancement_applied': 'transformer' in self.results and 'error' not in self.results['transformer'],
            'pipeline_version': "4.0",
            'processing_date': datetime.now().isoformat()
        }
        
        final_outputs['summary_stats'] = summary_stats
        return final_outputs
    
    def save_pipeline_results(self, results: Dict):
        """Save complete pipeline results"""
        results_file = self.output_path / "pipeline_results.json"
        
        # Convert Path objects to strings for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to: {results_file}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return f"Array/Tensor with shape {obj.shape}"
        else:
            return obj


def main():
    """Main function for running the unified pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gauge 3D Unified Pipeline")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Output directory (optional)")
    parser.add_argument("--config", "-c", help="Configuration file path (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize and run pipeline
        pipeline = UnifiedPipeline(args.config)
        results = pipeline.run_full_pipeline(args.input_video, args.output)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Input video: {results['input_video']}")
        print(f"Output directory: {results['output_directory']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Results saved to: {results['output_directory']}/pipeline_results.json")
        
        if 'summary_stats' in results['final_outputs']:
            stats = results['final_outputs']['summary_stats']
            print(f"\nSummary:")
            print(f"- Frames processed: {stats['total_frames_processed']}")
            print(f"- Gaussians generated: {stats['total_gaussians_generated']}")
            print(f"- Enhancement applied: {stats['enhancement_applied']}")
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
