#!/usr/bin/env python3
"""
Gauge 3D Visualization Demo

This script demonstrates how to use all the visualization tools in the Gauge 3D pipeline.
It provides examples of visualizing depth maps, point clouds, 4D Gaussians, and pipeline statistics.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import logging

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from viewer import DepthViewer, GaussianViewer, PointCloudViewer, PipelineViewer, StatsViewer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_depth_visualization():
    """Demonstrate depth visualization capabilities."""
    print("\n" + "="*60)
    print("DEPTH VISUALIZATION DEMO")
    print("="*60)
    
    depth_viewer = DepthViewer()
    
    # Find available depth datasets
    depth_datasets = depth_viewer.find_depth_datasets()
    print(f"Found {len(depth_datasets)} depth datasets: {depth_datasets}")
    
    if not depth_datasets:
        print("No depth datasets found. Skipping depth visualization demo.")
        return
    
    # Use the first available dataset
    dataset = depth_datasets[0]
    print(f"Using dataset: {dataset}")
    
    try:
        # Load depth maps
        depth_files = depth_viewer.find_depth_maps(dataset)
        if depth_files:
            print(f"Found {len(depth_files)} depth maps")
            
            # Visualize first few depth maps
            sample_files = depth_files[:3]  # First 3 frames
            fig = depth_viewer.visualize_depth_sequence(sample_files, 
                                                       title=f"Depth Maps - {dataset}")
            plt.show()
            
            # Create depth statistics plot
            stats_file = depth_viewer.depth_stats_dir / f"{dataset}_depth_results.json"
            if stats_file.exists():
                depth_stats = depth_viewer.load_depth_statistics(dataset)
                if depth_stats:
                    fig = depth_viewer.plot_depth_statistics(depth_stats)
                    plt.show()
        
        print("✓ Depth visualization demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in depth visualization demo: {e}")


def demo_gaussian_visualization():
    """Demonstrate 4D Gaussian visualization capabilities."""
    print("\n" + "="*60)
    print("4D GAUSSIAN VISUALIZATION DEMO")
    print("="*60)
    
    gaussian_viewer = GaussianViewer()
    
    # Find available Gaussian datasets
    gaussian_files = gaussian_viewer.find_gaussian_files()
    print(f"Found {len(gaussian_files)} Gaussian files")
    
    if not gaussian_files:
        print("No Gaussian files found. Skipping Gaussian visualization demo.")
        return
    
    try:
        # Load and visualize first few Gaussians
        sample_files = gaussian_files[:5]  # First 5 frames
        gaussian_data = []
        
        for file_path in sample_files:
            data = gaussian_viewer.load_gaussian_frame(file_path)
            if data:
                gaussian_data.append(data)
        
        if gaussian_data:
            print(f"Loaded {len(gaussian_data)} Gaussian frames")
            
            # Visualize single frame
            if gaussian_data:
                fig = gaussian_viewer.visualize_gaussian_frame(gaussian_data[0])
                plt.show()
            
            # Analyze temporal evolution
            if len(gaussian_data) > 1:
                stats = gaussian_viewer.analyze_temporal_evolution(gaussian_data)
                print("Temporal Evolution Statistics:")
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.2f}")
        
        print("✓ Gaussian visualization demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in Gaussian visualization demo: {e}")


def demo_point_cloud_visualization():
    """Demonstrate point cloud visualization capabilities."""
    print("\n" + "="*60)
    print("POINT CLOUD VISUALIZATION DEMO")
    print("="*60)
    
    pc_viewer = PointCloudViewer()
    
    # Find available point clouds
    point_clouds = pc_viewer.find_point_clouds()
    print(f"Found {len(point_clouds)} point cloud files")
    
    if not point_clouds:
        print("No point cloud files found. Skipping point cloud visualization demo.")
        return
    
    try:
        # Load and visualize first point cloud
        cloud_data = pc_viewer.load_point_cloud(point_clouds[0])
        if cloud_data and 'points' in cloud_data:
            points = cloud_data['points']
            colors = cloud_data.get('colors')
            
            print(f"Loaded point cloud with {len(points)} points")
            
            # Create matplotlib visualization
            fig = pc_viewer.visualize_point_cloud_matplotlib(points, colors, 
                                                           title=f"Point Cloud - {point_clouds[0].name}")
            plt.show()
            
            # Analyze statistics
            stats = pc_viewer.analyze_point_cloud_statistics(points)
            print("Point Cloud Statistics:")
            print(f"  Number of points: {stats['num_points']:,}")
            print(f"  Centroid: ({stats['centroid'][0]:.2f}, {stats['centroid'][1]:.2f}, {stats['centroid'][2]:.2f})")
            print(f"  Bounds: X[{stats['bounds']['min'][0]:.2f}, {stats['bounds']['max'][0]:.2f}]")
        
        print("✓ Point cloud visualization demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in point cloud visualization demo: {e}")


def demo_pipeline_overview():
    """Demonstrate pipeline overview visualization."""
    print("\n" + "="*60)
    print("PIPELINE OVERVIEW DEMO")
    print("="*60)
    
    pipeline_viewer = PipelineViewer()
    
    try:
        # Create comprehensive pipeline overview
        fig = pipeline_viewer.create_pipeline_overview()
        plt.show()
        
        # Generate pipeline report
        report = pipeline_viewer.generate_pipeline_report()
        print("\n" + report)
        
        print("✓ Pipeline overview demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline overview demo: {e}")


def demo_statistics_analysis():
    """Demonstrate comprehensive statistics analysis."""
    print("\n" + "="*60)
    print("STATISTICS ANALYSIS DEMO")
    print("="*60)
    
    stats_viewer = StatsViewer()
    
    try:
        # Load and analyze depth statistics
        depth_stats = stats_viewer.load_depth_statistics()
        if depth_stats:
            print("Analyzing depth statistics...")
            fig = stats_viewer.plot_depth_statistics(depth_stats)
            plt.show()
        
        # Load and analyze Gaussian statistics
        gaussian_stats = stats_viewer.load_gaussian_statistics()
        if gaussian_stats:
            print("Analyzing Gaussian statistics...")
            fig = stats_viewer.plot_gaussian_statistics(gaussian_stats)
            plt.show()
        
        # Generate comprehensive report
        report = stats_viewer.create_comprehensive_report()
        print("\n" + report)
        
        print("✓ Statistics analysis demo completed successfully")
        
    except Exception as e:
        logger.error(f"Error in statistics analysis demo: {e}")


def main():
    """Run all visualization demos."""
    print("Gauge 3D Visualization Suite Demo")
    print("This demo will showcase all visualization capabilities")
    print("=" * 60)
    
    # Check if output directory exists
    output_dir = Path("output")
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        print("Please run the pipeline first to generate visualization data.")
        return
    
    print(f"✓ Using output directory: {output_dir.absolute()}")
    
    # Run demos
    demos = [
        ("Pipeline Overview", demo_pipeline_overview),
        ("Statistics Analysis", demo_statistics_analysis),
        ("Depth Visualization", demo_depth_visualization),
        ("Gaussian Visualization", demo_gaussian_visualization),
        ("Point Cloud Visualization", demo_point_cloud_visualization),
    ]
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🚀 Starting {demo_name} demo...")
            demo_func()
        except KeyboardInterrupt:
            print(f"\n⏹️  {demo_name} demo interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ {demo_name} demo failed: {e}")
            continue
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("All visualization tools have been demonstrated.")
    print("You can now use these tools to analyze your pipeline outputs!")


if __name__ == "__main__":
    main()
