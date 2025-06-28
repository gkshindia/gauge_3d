#!/usr/bin/env python3
"""
Run End-to-End Pipeline from Existing Depth Maps

This script runs the complete Gauge 3D pipeline using existing depth estimation output,
configured for standard quality (100K-500K points per frame, 50K-200K Gaussians per frame).
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from unified_pipeline import UnifiedPipeline, PipelineError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('end_to_end_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EndToEndRunner:
    """Runner for end-to-end pipeline with existing depth maps"""
    
    def __init__(self):
        self.pipeline = UnifiedPipeline("config/pipeline_config.yaml")
        self.input_video_path = Path("vinput/1080_60_fps.mp4")
        self.depth_maps_path = Path("output/depth_maps/1080_60_fps")
        self.output_path = Path("output")
        
    def run_pipeline(self):
        """Run the complete end-to-end pipeline"""
        logger.info("Starting end-to-end pipeline execution...")
        logger.info(f"Using existing depth maps from: {self.depth_maps_path}")
        logger.info(f"Output will be saved to: {self.output_path}")
        
        # Check if depth maps exist
        if not self.depth_maps_path.exists():
            raise PipelineError(f"Depth maps directory not found: {self.depth_maps_path}")
        
        depth_files = list(self.depth_maps_path.glob("*.npy"))
        logger.info(f"Found {len(depth_files)} depth map files")
        
        if len(depth_files) == 0:
            raise PipelineError("No depth map files found")
        
        try:
            # Run the pipeline - it will use existing depth maps if they exist
            results = self.pipeline.run_full_pipeline(
                input_video=self.input_video_path,
                output_dir=self.output_path
            )
            
            logger.info("Pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main execution function"""
    print("="*80)
    print("GAUGE 3D END-TO-END PIPELINE")
    print("Standard Quality Configuration")
    print("- Point clouds: 100K-500K points per frame")
    print("- Gaussians: 50K-200K per frame")
    print("="*80)
    
    try:
        runner = EndToEndRunner()
        start_time = time.time()
        
        results = runner.run_pipeline()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED!")
        print("="*80)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Input video: {results['input_video']}")
        print(f"Output directory: {results['output_directory']}")
         if 'summary_stats' in results['final_outputs']:
            stats = results['final_outputs']['summary_stats']
            print("\nResults Summary:")
            print(f"- Total frames processed: {stats['total_frames_processed']}")
            print(f"- Total Gaussians generated: {stats['total_gaussians_generated']}")
            print(f"- Enhancement applied: {stats['enhancement_applied']}")
            print(f"- Pipeline version: {stats['pipeline_version']}")
        
        print("\nOutput files:")
        for output_name, output_path in results['final_outputs'].items():
            if isinstance(output_path, (str, Path)) and output_name != 'summary_stats':
                print(f"- {output_name}: {output_path}")
        
        print(f"\nDetailed results saved to: {results['output_directory']}/pipeline_results.json")
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
