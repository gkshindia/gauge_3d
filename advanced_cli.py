#!/usr/bin/env python3
"""
Gauge 3D Enhanced Command Line Interface

Comprehensive CLI supporting the full pipeline with advanced features:
- Full pipeline execution with configuration support
- Batch processing capabilities
- Pipeline resumption and checkpointing
- Interactive mode for parameter tuning
- Visualization and results analysis
"""

import sys
import argparse
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from unified_pipeline import UnifiedPipeline, PipelineError
from distance_measurement import DistanceMeasurementTool
from performance_optimizer import PerformanceProfiler
from validation_suite import ValidationSuite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLIError(Exception):
    """Custom exception for CLI errors"""
    pass


class AdvancedCLI:
    """Advanced command line interface for Gauge 3D pipeline"""
    
    def __init__(self):
        self.pipeline = None
        self.config_path = None
        self.output_base = Path("output")
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser"""
        parser = argparse.ArgumentParser(
            description="Gauge 3D: Advanced 3D Reconstruction and Analysis Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run full pipeline on a video
  python advanced_cli.py process video.mp4 --output results/
  
  # Batch process multiple videos
  python advanced_cli.py batch videos/ --pattern "*.mp4" --output batch_results/
  
  # Resume from checkpoint
  python advanced_cli.py resume checkpoint.json --continue-from gaussian
  
  # Run with custom configuration
  python advanced_cli.py process video.mp4 --config custom_config.yaml
  
  # Interactive mode
  python advanced_cli.py interactive
  
  # Measure distances in processed results
  python advanced_cli.py measure results/enhanced/ --point1 100,200,50 --point2 150,250,75
  
  # Validate pipeline results
  python advanced_cli.py validate results/ --ground-truth truth.json
  
  # Performance analysis
  python advanced_cli.py profile results/ --benchmark
            """
        )
        
        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Process command (main pipeline)
        self._add_process_command(subparsers)
        
        # Batch processing command
        self._add_batch_command(subparsers)
        
        # Resume command
        self._add_resume_command(subparsers)
        
        # Interactive command
        self._add_interactive_command(subparsers)
        
        # Measurement command
        self._add_measure_command(subparsers)
        
        # Validation command
        self._add_validate_command(subparsers)
        
        # Profile command
        self._add_profile_command(subparsers)
        
        # Visualize command
        self._add_visualize_command(subparsers)
        
        # Configuration command
        self._add_config_command(subparsers)
        
        return parser
    
    def _add_process_command(self, subparsers):
        """Add process command for single video processing"""
        process_parser = subparsers.add_parser('process', help='Process a single video file')
        process_parser.add_argument('input_video', help='Path to input video file')
        process_parser.add_argument('--output', '-o', help='Output directory', default='output')
        process_parser.add_argument('--config', '-c', help='Configuration file path')
        process_parser.add_argument('--phases', choices=['depth', 'gaussian', 'transformer', 'all'], 
                                  default='all', help='Phases to run')
        process_parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
        process_parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
        process_parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
        process_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
        process_parser.add_argument('--checkpoint', help='Save checkpoint to file')
        
    def _add_batch_command(self, subparsers):
        """Add batch processing command"""
        batch_parser = subparsers.add_parser('batch', help='Batch process multiple videos')
        batch_parser.add_argument('input_dir', help='Directory containing input videos')
        batch_parser.add_argument('--pattern', default='*.mp4', help='File pattern to match')
        batch_parser.add_argument('--output', '-o', help='Output base directory', default='batch_output')
        batch_parser.add_argument('--config', '-c', help='Configuration file path')
        batch_parser.add_argument('--parallel', '-p', type=int, help='Number of parallel processes')
        batch_parser.add_argument('--continue-on-error', action='store_true', help='Continue on individual failures')
        batch_parser.add_argument('--phases', choices=['depth', 'gaussian', 'transformer', 'all'], 
                                default='all', help='Phases to run')
        
    def _add_resume_command(self, subparsers):
        """Add resume command for checkpoint resumption"""
        resume_parser = subparsers.add_parser('resume', help='Resume from checkpoint')
        resume_parser.add_argument('checkpoint_file', help='Path to checkpoint file')
        resume_parser.add_argument('--continue-from', choices=['depth', 'gaussian', 'transformer'], 
                                 help='Phase to continue from')
        resume_parser.add_argument('--output', '-o', help='Override output directory')
        
    def _add_interactive_command(self, subparsers):
        """Add interactive mode command"""
        interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
        interactive_parser.add_argument('--config', '-c', help='Initial configuration file')
        
    def _add_measure_command(self, subparsers):
        """Add measurement command"""
        measure_parser = subparsers.add_parser('measure', help='Measure distances in 3D data')
        measure_parser.add_argument('data_dir', help='Directory containing processed 3D data')
        measure_parser.add_argument('--point1', help='First point coordinates (x,y,z)')
        measure_parser.add_argument('--point2', help='Second point coordinates (x,y,z)')
        measure_parser.add_argument('--interactive', action='store_true', help='Interactive point selection')
        measure_parser.add_argument('--output', '-o', help='Output file for measurements')
        measure_parser.add_argument('--visualize', action='store_true', help='Show 3D visualization')
        
    def _add_validate_command(self, subparsers):
        """Add validation command"""
        validate_parser = subparsers.add_parser('validate', help='Validate pipeline results')
        validate_parser.add_argument('results_dir', help='Directory containing pipeline results')
        validate_parser.add_argument('--ground-truth', help='Ground truth data file')
        validate_parser.add_argument('--metrics', nargs='+', 
                                   choices=['depth_accuracy', 'reconstruction_quality', 'measurement_precision'],
                                   default=['depth_accuracy', 'reconstruction_quality'],
                                   help='Validation metrics to compute')
        validate_parser.add_argument('--report', help='Generate validation report file')
        
    def _add_profile_command(self, subparsers):
        """Add profiling command"""
        profile_parser = subparsers.add_parser('profile', help='Performance analysis and profiling')
        profile_parser.add_argument('results_dir', help='Directory containing pipeline results')
        profile_parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
        profile_parser.add_argument('--memory', action='store_true', help='Analyze memory usage')
        profile_parser.add_argument('--gpu', action='store_true', help='Analyze GPU utilization')
        profile_parser.add_argument('--report', help='Generate performance report file')
        
    def _add_visualize_command(self, subparsers):
        """Add visualization command"""
        visualize_parser = subparsers.add_parser('visualize', help='Visualize pipeline results')
        visualize_parser.add_argument('results_dir', help='Directory containing pipeline results')
        visualize_parser.add_argument('--type', choices=['depth', 'point_cloud', '4d_gaussian', 'measurements'],
                                    default='point_cloud', help='Visualization type')
        visualize_parser.add_argument('--frame', type=int, help='Specific frame to visualize')
        visualize_parser.add_argument('--interactive', action='store_true', help='Interactive 3D viewer')
        visualize_parser.add_argument('--save', help='Save visualization to file')
        
    def _add_config_command(self, subparsers):
        """Add configuration management command"""
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_action', help='Configuration actions')
        
        # Create default config
        create_parser = config_subparsers.add_parser('create', help='Create default configuration')
        create_parser.add_argument('output_file', help='Output configuration file')
        
        # Validate config
        validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
        validate_parser.add_argument('config_file', help='Configuration file to validate')
        
        # Show config
        show_parser = config_subparsers.add_parser('show', help='Show current configuration')
        show_parser.add_argument('config_file', nargs='?', help='Configuration file to show')
    
    def run(self, args=None):
        """Main CLI execution"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return
        
        try:
            # Set up logging level
            if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
                logging.getLogger().setLevel(logging.DEBUG)
            
            # Route to appropriate command handler
            command_handlers = {
                'process': self.handle_process,
                'batch': self.handle_batch,
                'resume': self.handle_resume,
                'interactive': self.handle_interactive,
                'measure': self.handle_measure,
                'validate': self.handle_validate,
                'profile': self.handle_profile,
                'visualize': self.handle_visualize,
                'config': self.handle_config
            }
            
            handler = command_handlers.get(parsed_args.command)
            if handler:
                return handler(parsed_args)
            else:
                raise CLIError(f"Unknown command: {parsed_args.command}")
                
        except KeyboardInterrupt:
            print("\\nOperation cancelled by user")
            return 1
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def handle_process(self, args) -> int:
        """Handle single video processing"""
        print(f"Processing video: {args.input_video}")
        
        try:
            # Initialize pipeline
            pipeline = UnifiedPipeline(args.config)
            
            # Configure GPU/CPU usage
            if args.gpu and args.cpu:
                raise CLIError("Cannot specify both --gpu and --cpu")
            
            if args.gpu:
                pipeline.config['pipeline']['enable_gpu'] = True
            elif args.cpu:
                pipeline.config['pipeline']['enable_gpu'] = False
            
            # Set max frames if specified
            if args.max_frames:
                pipeline.config['pipeline']['max_frames'] = args.max_frames
            
            # Run pipeline
            results = pipeline.run_full_pipeline(args.input_video, args.output)
            
            # Save checkpoint if requested
            if args.checkpoint:
                self._save_checkpoint(results, args.checkpoint)
            
            print(f"\\n✅ Processing completed successfully!")
            print(f"Output saved to: {results['output_directory']}")
            print(f"Processing time: {results['processing_time']:.2f} seconds")
            
            return 0
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return 1
    
    def handle_batch(self, args) -> int:
        """Handle batch processing"""
        print(f"Batch processing videos from: {args.input_dir}")
        
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise CLIError(f"Input directory not found: {input_dir}")
        
        # Find video files
        video_files = list(input_dir.glob(args.pattern))
        if not video_files:
            raise CLIError(f"No videos found matching pattern: {args.pattern}")
        
        print(f"Found {len(video_files)} video(s) to process")
        
        success_count = 0
        results = []
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\\n[{i}/{len(video_files)}] Processing: {video_file.name}")
            
            try:
                # Create output directory for this video
                output_dir = Path(args.output) / video_file.stem
                
                # Process video
                pipeline = UnifiedPipeline(args.config)
                result = pipeline.run_full_pipeline(video_file, output_dir)
                
                results.append({
                    'video_file': str(video_file),
                    'success': True,
                    'output_dir': str(output_dir),
                    'processing_time': result['processing_time']
                })
                
                success_count += 1
                print(f"✅ {video_file.name} completed in {result['processing_time']:.2f}s")
                
            except Exception as e:
                error_msg = f"Failed to process {video_file.name}: {e}"
                logger.error(error_msg)
                
                results.append({
                    'video_file': str(video_file),
                    'success': False,
                    'error': str(e)
                })
                
                if not args.continue_on_error:
                    print(f"❌ Batch processing stopped due to error")
                    break
                else:
                    print(f"⚠️  Continuing with next video...")
        
        # Save batch results
        batch_results_file = Path(args.output) / "batch_results.json"
        batch_results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(batch_results_file, 'w') as f:
            json.dump({
                'total_videos': len(video_files),
                'successful': success_count,
                'failed': len(video_files) - success_count,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\\n📊 Batch processing completed:")
        print(f"  Successful: {success_count}/{len(video_files)}")
        print(f"  Results saved to: {batch_results_file}")
        
        return 0 if success_count == len(video_files) else 1
    
    def handle_resume(self, args) -> int:
        """Handle pipeline resumption from checkpoint"""
        print(f"Resuming from checkpoint: {args.checkpoint_file}")
        
        checkpoint_file = Path(args.checkpoint_file)
        if not checkpoint_file.exists():
            raise CLIError(f"Checkpoint file not found: {checkpoint_file}")
        
        # Load checkpoint
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        # Resume processing
        # This would require implementing checkpoint resumption in UnifiedPipeline
        print("⚠️  Checkpoint resumption not yet implemented")
        return 1
    
    def handle_interactive(self, args) -> int:
        """Handle interactive mode"""
        print("🎮 Starting interactive mode...")
        print("This feature will be implemented in a future version")
        return 0
    
    def handle_measure(self, args) -> int:
        """Handle distance measurement"""
        print(f"Measuring distances in: {args.data_dir}")
        
        try:
            measurement_tool = DistanceMeasurementTool()
            
            if args.interactive:
                # Interactive point selection
                result = measurement_tool.interactive_measurement(args.data_dir)
            elif args.point1 and args.point2:
                # Manual point specification
                point1 = [float(x) for x in args.point1.split(',')]
                point2 = [float(x) for x in args.point2.split(',')]
                result = measurement_tool.measure_distance(args.data_dir, point1, point2)
            else:
                raise CLIError("Must specify either --interactive or both --point1 and --point2")
            
            print(f"✅ Distance measurement: {result['distance']:.3f} units")
            
            if args.output:
                measurement_tool.export_measurement(result, args.output)
                print(f"📁 Results saved to: {args.output}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Measurement failed: {e}")
            return 1
    
    def handle_validate(self, args) -> int:
        """Handle pipeline validation"""
        print(f"Validating results in: {args.results_dir}")
        
        try:
            validator = ValidationSuite()
            
            validation_results = validator.validate_pipeline_results(
                args.results_dir,
                ground_truth_file=args.ground_truth,
                metrics=args.metrics
            )
            
            print("✅ Validation completed")
            for metric, score in validation_results.items():
                print(f"  {metric}: {score:.3f}")
            
            if args.report:
                validator.generate_validation_report(validation_results, args.report)
                print(f"📄 Report saved to: {args.report}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return 1
    
    def handle_profile(self, args) -> int:
        """Handle performance profiling"""
        print(f"Profiling results in: {args.results_dir}")
        
        try:
            profiler = PerformanceProfiler()
            
            profile_results = profiler.analyze_pipeline_performance(
                args.results_dir,
                include_memory=args.memory,
                include_gpu=args.gpu,
                run_benchmark=args.benchmark
            )
            
            print("✅ Profiling completed")
            print(f"  Total processing time: {profile_results.get('total_time', 0):.2f}s")
            print(f"  Peak memory usage: {profile_results.get('peak_memory', 0):.1f}MB")
            
            if args.report:
                profiler.generate_performance_report(profile_results, args.report)
                print(f"📄 Report saved to: {args.report}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            return 1
    
    def handle_visualize(self, args) -> int:
        """Handle visualization"""
        print(f"Visualizing {args.type} data from: {args.results_dir}")
        
        try:
            from viewer.pipeline_viewer import PipelineViewer
            
            viewer = PipelineViewer()
            
            if args.interactive:
                viewer.show_interactive_3d(args.results_dir, args.type, args.frame)
            else:
                viewer.generate_static_visualization(args.results_dir, args.type, args.frame, args.save)
            
            print("✅ Visualization completed")
            return 0
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return 1
    
    def handle_config(self, args) -> int:
        """Handle configuration management"""
        if args.config_action == 'create':
            return self._create_default_config(args.output_file)
        elif args.config_action == 'validate':
            return self._validate_config(args.config_file)
        elif args.config_action == 'show':
            return self._show_config(args.config_file)
        else:
            print("Available config actions: create, validate, show")
            return 1
    
    def _create_default_config(self, output_file: str) -> int:
        """Create default configuration file"""
        try:
            from unified_pipeline import ConfigurationManager
            
            config_manager = ConfigurationManager()
            config_file = Path(output_file)
            
            # Save default config
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(config_manager.config, f, default_flow_style=False)
            
            print(f"✅ Default configuration created: {config_file}")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to create config: {e}")
            return 1
    
    def _validate_config(self, config_file: str) -> int:
        """Validate configuration file"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                raise CLIError(f"Config file not found: {config_file}")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            required_sections = ['pipeline', 'depth_estimation', 'gaussian_4d', 'transformer']
            for section in required_sections:
                if section not in config:
                    raise CLIError(f"Missing required config section: {section}")
            
            print(f"✅ Configuration is valid: {config_file}")
            return 0
            
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return 1
    
    def _show_config(self, config_file: Optional[str]) -> int:
        """Show configuration"""
        try:
            if config_file:
                config_path = Path(config_file)
                if not config_path.exists():
                    raise CLIError(f"Config file not found: {config_file}")
                
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                from unified_pipeline import ConfigurationManager
                config_manager = ConfigurationManager()
                config = config_manager.config
            
            print("📋 Current Configuration:")
            print(yaml.dump(config, default_flow_style=False))
            return 0
            
        except Exception as e:
            logger.error(f"Failed to show config: {e}")
            return 1
    
    def _save_checkpoint(self, results: Dict, checkpoint_file: str):
        """Save pipeline checkpoint"""
        checkpoint_path = Path(checkpoint_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'version': '4.0'
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved to: {checkpoint_path}")


def main():
    """Main CLI entry point"""
    cli = AdvancedCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
