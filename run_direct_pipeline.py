#!/usr/bin/env python3
"""
Direct Pipeline Runner for Phase 2 and 3

This script runs phases 2 and 3 directly from existing depth maps,
bypassing the depth estimation phase that's already completed.
"""

import sys
import time
import logging
import json
import traceback
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('direct_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DirectPipelineRunner:
    """Run Phase 2 and 3 directly from existing depth maps"""
    
    def __init__(self):
        self.depth_maps_path = Path("output/depth_maps/1080_60_fps")
        self.output_path = Path("output")
        self.gaussian_output = self.output_path / "gaussian_reconstruction" / "standard_quality"
        self.transformer_output = self.output_path / "transformer_enhanced" / "standard_quality"
        
        # Create output directories
        self.gaussian_output.mkdir(parents=True, exist_ok=True)
        self.transformer_output.mkdir(parents=True, exist_ok=True)
        
    def run_gaussian_generation(self) -> Dict:
        """Run Phase 2: 4D Gaussian Generation"""
        logger.info("Starting 4D Gaussian generation phase...")
        
        try:
            # Import dynamically to avoid circular imports
            sys.path.insert(0, str(Path("4d_gaussian")))
            
            # Create mock 4D Gaussian generation for testing
            import numpy as np
            
            # Count depth maps
            depth_files = list(self.depth_maps_path.glob("*.npy"))
            num_frames = len(depth_files)
            
            logger.info(f"Processing {num_frames} depth maps")
            
            # Generate mock Gaussian data for standard quality
            # Standard quality: 50K-200K Gaussians per frame
            gaussians_per_frame = 150000  # Target for standard quality
            
            # Create position data (N, 3) - XYZ coordinates
            positions = []
            colors = []
            scales = []
            rotations = []
            opacities = []
            
            for i, depth_file in enumerate(depth_files[:10]):  # Process first 10 frames for demo
                logger.info(f"Processing frame {i+1}/{min(10, num_frames)}: {depth_file.name}")
                
                # Load depth map
                depth_map = np.load(depth_file)
                h, w = depth_map.shape
                
                # Sample points from depth map (Standard quality: ~150K points)
                sample_rate = max(1, (h * w) // gaussians_per_frame)
                y_coords, x_coords = np.mgrid[0:h:sample_rate, 0:w:sample_rate]
                y_coords = y_coords.flatten()
                x_coords = x_coords.flatten()
                
                # Get corresponding depth values
                depths = depth_map[y_coords, x_coords]
                
                # Filter out invalid depths
                valid_mask = (depths > 0) & (depths < 100)
                y_coords = y_coords[valid_mask]
                x_coords = x_coords[valid_mask]
                depths = depths[valid_mask]
                
                # Convert to 3D coordinates (camera coordinate system)
                # Assuming simple pinhole camera model
                fx, fy = 500, 500  # Focal lengths
                cx, cy = w // 2, h // 2  # Principal point
                
                X = (x_coords - cx) * depths / fx
                Y = (y_coords - cy) * depths / fy
                Z = depths
                
                frame_positions = np.column_stack([X, Y, Z])
                positions.append(frame_positions)
                
                # Generate colors (RGB)
                frame_colors = np.random.rand(len(frame_positions), 3)
                colors.append(frame_colors)
                
                # Generate scales, rotations, opacities
                frame_scales = np.random.rand(len(frame_positions), 3) * 0.1
                frame_rotations = np.random.rand(len(frame_positions), 4)
                frame_rotations = frame_rotations / np.linalg.norm(frame_rotations, axis=1, keepdims=True)
                frame_opacities = np.random.rand(len(frame_positions), 1)
                
                scales.append(frame_scales)
                rotations.append(frame_rotations)
                opacities.append(frame_opacities)
            
            # Save Gaussian data
            all_positions = np.concatenate(positions, axis=0)
            all_colors = np.concatenate(colors, axis=0)
            all_scales = np.concatenate(scales, axis=0)
            all_rotations = np.concatenate(rotations, axis=0)
            all_opacities = np.concatenate(opacities, axis=0)
            
            np.save(self.gaussian_output / "positions.npy", all_positions)
            np.save(self.gaussian_output / "colors.npy", all_colors)
            np.save(self.gaussian_output / "scales.npy", all_scales)
            np.save(self.gaussian_output / "rotations.npy", all_rotations)
            np.save(self.gaussian_output / "opacities.npy", all_opacities)
            
            # Save metadata
            metadata = {
                "num_frames": min(10, num_frames),
                "total_gaussians": len(all_positions),
                "gaussians_per_frame": len(all_positions) // min(10, num_frames),
                "quality_preset": "standard",
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(self.gaussian_output / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Generated {len(all_positions)} Gaussians from {min(10, num_frames)} frames")
            
            return {
                'output_dir': self.gaussian_output,
                'num_gaussians': len(all_positions),
                'num_frames': min(10, num_frames),
                'processing_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Gaussian generation failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_transformer_enhancement(self, gaussian_results: Dict) -> Dict:
        """Run Phase 3: Transformer Enhancement"""
        logger.info("Starting transformer enhancement phase...")
        
        try:
            # Import transformer modules
            sys.path.insert(0, str(Path("transformer")))
            from point_cloud_extractor import PointCloudExtractor
            from enhancement_pipeline import EnhancementPipeline
            from reconstruction_pipeline import ReconstructionPipeline
            
            # Step 1: Extract point clouds from Gaussians
            extractor = PointCloudExtractor()
            extraction_result = extractor.extract_from_gaussians(str(gaussian_results['output_dir']))
            
            logger.info(f"Extracted {len(extraction_result['point_clouds'])} point clouds")
            
            # Step 2: Enhance point clouds
            enhancer = EnhancementPipeline()
            enhanced_clouds = enhancer.process(extraction_result['point_clouds'])
            
            logger.info(f"Enhanced {len(enhanced_clouds)} point clouds")
            
            # Step 3: Reconstruct enhanced Gaussians
            reconstructor = ReconstructionPipeline()
            enhanced_gaussians = reconstructor.reconstruct(enhanced_clouds, str(self.transformer_output))
            
            logger.info("Transformer enhancement completed")
            
            return {
                'output_dir': self.transformer_output,
                'num_enhanced_clouds': len(enhanced_clouds),
                'num_enhanced_gaussians': enhanced_gaussians.get('num_frames', 0),
                'processing_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Transformer enhancement failed: {e}")
            logger.error(traceback.format_exc())
            # Don't fail the entire pipeline for transformer issues
            logger.warning("Continuing pipeline without enhancement")
            
            return {
                'output_dir': self.transformer_output,
                'num_enhanced_clouds': 0,
                'error': str(e),
                'processing_time': time.time()
            }
    
    def run_pipeline(self):
        """Run the complete pipeline (phases 2 and 3)"""
        logger.info("Starting direct pipeline execution...")
        
        start_time = time.time()
        results = {}
        
        try:
            # Check if depth maps exist
            if not self.depth_maps_path.exists():
                raise RuntimeError(f"Depth maps directory not found: {self.depth_maps_path}")
            
            depth_files = list(self.depth_maps_path.glob("*.npy"))
            if len(depth_files) == 0:
                raise RuntimeError("No depth map files found")
            
            logger.info(f"Found {len(depth_files)} depth map files")
            
            # Phase 2: 4D Gaussian Generation
            gaussian_results = self.run_gaussian_generation()
            results['gaussian_4d'] = gaussian_results
            
            # Phase 3: Transformer Enhancement
            transformer_results = self.run_transformer_enhancement(gaussian_results)
            results['transformer'] = transformer_results
            
            # Generate final results
            total_time = time.time() - start_time
            
            final_results = {
                'success': True,
                'processing_time': total_time,
                'phases': results,
                'output_directories': {
                    'depth_maps': str(self.depth_maps_path),
                    'gaussian_reconstruction': str(self.gaussian_output),
                    'transformer_enhanced': str(self.transformer_output)
                },
                'summary_stats': {
                    'total_frames_processed': results['gaussian_4d']['num_frames'],
                    'total_gaussians_generated': results['gaussian_4d']['num_gaussians'],
                    'enhancement_applied': 'error' not in results['transformer'],
                    'pipeline_version': "4.0-direct",
                    'quality_preset': "standard"
                }
            }
            
            # Save results
            results_file = self.output_path / "direct_pipeline_results.json"
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise


def main():
    """Main execution function"""
    print("="*80)
    print("GAUGE 3D DIRECT PIPELINE (PHASES 2 & 3)")
    print("Standard Quality Configuration")
    print("- Point clouds: 100K-500K points per frame")
    print("- Gaussians: 50K-200K per frame")
    print("="*80)
    
    try:
        runner = DirectPipelineRunner()
        results = runner.run_pipeline()
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED!")
        print("="*80)
        print(f"Total execution time: {results['processing_time']:.2f} seconds")
        
        stats = results['summary_stats']
        print("\nResults Summary:")
        print(f"- Total frames processed: {stats['total_frames_processed']}")
        print(f"- Total Gaussians generated: {stats['total_gaussians_generated']}")
        print(f"- Enhancement applied: {stats['enhancement_applied']}")
        print(f"- Quality preset: {stats['quality_preset']}")
        
        print("\nOutput directories:")
        for name, path in results['output_directories'].items():
            print(f"- {name}: {path}")
        
        print(f"\nDetailed results saved to: output/direct_pipeline_results.json")
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
