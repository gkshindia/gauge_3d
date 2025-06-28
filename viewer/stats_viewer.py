"""
Statistics and Metrics Visualization Module

Provides comprehensive analysis and visualization of pipeline statistics,
performance metrics, and quality assessments.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from datetime import datetime
import seaborn as sns
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import seaborn as sns
    import pandas as pd
    HAS_SEABORN = True
    # Set seaborn style for better plots
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False


class StatsViewer:
    """Comprehensive statistics visualization and analysis tools."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the StatsViewer.
        
        Args:
            output_dir: Base output directory containing pipeline outputs
        """
        self.output_dir = Path(output_dir)
        self.stats_dir = self.output_dir / "depth_stats"
        self.gaussian_dir = self.output_dir / "gaussian_reconstruction"
        self.logs_dir = self.gaussian_dir / "logs"
    
    def load_depth_statistics(self, dataset_name: Optional[str] = None) -> Optional[Dict]:
        """
        Load depth estimation statistics.
        
        Args:
            dataset_name: Specific dataset name or None for auto-detection
            
        Returns:
            Dictionary containing depth statistics or None if not found
        """
        if not self.stats_dir.exists():
            logging.warning(f"Stats directory not found: {self.stats_dir}")
            return None
        
        # Find stats files
        if dataset_name:
            stats_file = self.stats_dir / f"{dataset_name}_depth_results.json"
        else:
            # Auto-detect the first available stats file
            stats_files = list(self.stats_dir.glob("*_depth_results.json"))
            if not stats_files:
                logging.warning("No depth statistics files found")
                return None
            stats_file = stats_files[0]
        
        try:
            with open(stats_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load depth statistics: {e}")
            return None
    
    def load_gaussian_statistics(self, dataset_name: Optional[str] = None) -> Optional[Dict]:
        """
        Load 4D Gaussian reconstruction statistics.
        
        Args:
            dataset_name: Specific dataset name or None for auto-detection
            
        Returns:
            Dictionary containing Gaussian statistics or None if not found
        """
        if not self.gaussian_dir.exists():
            logging.warning(f"Gaussian directory not found: {self.gaussian_dir}")
            return None
        
        # Find Gaussian data files
        if dataset_name:
            gaussian_file = self.gaussian_dir / f"{dataset_name}_gaussian_data.json"
        else:
            # Auto-detect the first available Gaussian file
            gaussian_files = list(self.gaussian_dir.glob("*_gaussian_data.json"))
            if not gaussian_files:
                logging.warning("No Gaussian statistics files found")
                return None
            gaussian_file = gaussian_files[0]
        
        try:
            with open(gaussian_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load Gaussian statistics: {e}")
            return None
    
    def analyze_depth_quality(self, depth_stats: Dict) -> Dict:
        """
        Analyze depth estimation quality metrics.
        
        Args:
            depth_stats: Depth statistics dictionary
            
        Returns:
            Quality analysis results
        """
        quality_metrics = {}
        
        # Basic depth metrics
        if 'mean_depth' in depth_stats and 'std_depth' in depth_stats:
            quality_metrics['depth_consistency'] = depth_stats['std_depth'] / depth_stats['mean_depth']
            quality_metrics['depth_range'] = depth_stats.get('max_depth', 0) - depth_stats.get('min_depth', 0)
        
        # Temporal consistency (if frame-by-frame stats available)
        if 'frame_stats' in depth_stats:
            frame_means = [frame['mean_depth'] for frame in depth_stats['frame_stats']]
            frame_stds = [frame['std_depth'] for frame in depth_stats['frame_stats']]
            
            quality_metrics['temporal_consistency'] = {
                'mean_variation': np.std(frame_means),
                'std_variation': np.std(frame_stds),
                'trend_slope': np.polyfit(range(len(frame_means)), frame_means, 1)[0] if len(frame_means) > 1 else 0
            }
        
        # Coverage metrics
        if 'coverage_percentage' in depth_stats:
            quality_metrics['coverage'] = depth_stats['coverage_percentage']
        
        return quality_metrics
    
    def analyze_gaussian_quality(self, gaussian_stats: Dict) -> Dict:
        """
        Analyze 4D Gaussian reconstruction quality metrics.
        
        Args:
            gaussian_stats: Gaussian statistics dictionary
            
        Returns:
            Quality analysis results
        """
        quality_metrics = {}
        
        # Gaussian count statistics
        if 'total_gaussians' in gaussian_stats:
            quality_metrics['gaussian_density'] = gaussian_stats['total_gaussians']
        
        # Frame-by-frame analysis
        if 'frame_gaussian_counts' in gaussian_stats:
            counts = gaussian_stats['frame_gaussian_counts']
            quality_metrics['gaussian_consistency'] = {
                'mean_count': np.mean(counts),
                'std_count': np.std(counts),
                'min_count': np.min(counts),
                'max_count': np.max(counts),
                'coefficient_of_variation': np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
            }
        
        # Spatial distribution
        if 'spatial_bounds' in gaussian_stats:
            bounds = gaussian_stats['spatial_bounds']
            quality_metrics['spatial_coverage'] = {
                'x_range': bounds['max_x'] - bounds['min_x'],
                'y_range': bounds['max_y'] - bounds['min_y'],
                'z_range': bounds['max_z'] - bounds['min_z'],
                'volume': (bounds['max_x'] - bounds['min_x']) * 
                         (bounds['max_y'] - bounds['min_y']) * 
                         (bounds['max_z'] - bounds['min_z'])
            }
        
        return quality_metrics
    
    def plot_depth_statistics(self, depth_stats: Dict, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create comprehensive depth statistics visualization.
        
        Args:
            depth_stats: Depth statistics dictionary
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Depth Estimation Statistics Analysis', fontsize=16, fontweight='bold')
        
        # 1. Basic depth metrics
        ax1 = axes[0, 0]
        metrics = ['mean_depth', 'std_depth', 'min_depth', 'max_depth']
        values = [depth_stats.get(metric, 0) for metric in metrics]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_title('Basic Depth Metrics')
        ax1.set_ylabel('Depth Value')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 2. Depth distribution histogram
        ax2 = axes[0, 1]
        if 'depth_histogram' in depth_stats:
            hist_data = depth_stats['depth_histogram']
            ax2.bar(hist_data['bins'][:-1], hist_data['counts'], 
                   width=np.diff(hist_data['bins']), alpha=0.7, color='steelblue')
            ax2.set_title('Depth Value Distribution')
            ax2.set_xlabel('Depth Value')
            ax2.set_ylabel('Frequency')
        else:
            ax2.text(0.5, 0.5, 'Histogram data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Depth Value Distribution')
        
        # 3. Coverage statistics
        ax3 = axes[0, 2]
        coverage_metrics = ['coverage_percentage', 'valid_pixels', 'invalid_pixels']
        coverage_values = [depth_stats.get(metric, 0) for metric in coverage_metrics]
        
        if any(coverage_values):
            ax3.pie(coverage_values, labels=coverage_metrics, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Depth Coverage')
        else:
            ax3.text(0.5, 0.5, 'Coverage data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Depth Coverage')
        
        # 4. Frame-by-frame analysis
        ax4 = axes[1, 0]
        if 'frame_stats' in depth_stats:
            frame_stats = depth_stats['frame_stats']
            frame_nums = range(len(frame_stats))
            frame_means = [fs['mean_depth'] for fs in frame_stats]
            frame_stds = [fs['std_depth'] for fs in frame_stats]
            
            ax4_twin = ax4.twinx()
            line1 = ax4.plot(frame_nums, frame_means, 'b-', label='Mean Depth', alpha=0.7)
            line2 = ax4_twin.plot(frame_nums, frame_stds, 'r-', label='Std Depth', alpha=0.7)
            
            ax4.set_xlabel('Frame Number')
            ax4.set_ylabel('Mean Depth', color='b')
            ax4_twin.set_ylabel('Std Depth', color='r')
            ax4.set_title('Temporal Depth Evolution')
            
            # Combine legends
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax4.text(0.5, 0.5, 'Frame-by-frame data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Temporal Depth Evolution')
        
        # 5. Quality metrics
        ax5 = axes[1, 1]
        quality_metrics = self.analyze_depth_quality(depth_stats)
        
        if quality_metrics:
            metric_names = []
            metric_values = []
            
            for key, value in quality_metrics.items():
                if isinstance(value, (int, float)):
                    metric_names.append(key.replace('_', ' ').title())
                    metric_values.append(value)
            
            if metric_names:
                bars = ax5.barh(metric_names, metric_values, color='lightblue', alpha=0.7)
                ax5.set_title('Quality Metrics')
                ax5.set_xlabel('Metric Value')
                
                # Add value labels
                for bar, value in zip(bars, metric_values):
                    ax5.text(bar.get_width() + max(metric_values)*0.01, 
                            bar.get_y() + bar.get_height()/2,
                            f'{value:.3f}', ha='left', va='center')
        else:
            ax5.text(0.5, 0.5, 'Quality metrics not available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Quality Metrics')
        
        # 6. Processing performance
        ax6 = axes[1, 2]
        perf_metrics = ['processing_time', 'frames_per_second', 'total_frames']
        perf_values = [depth_stats.get(metric, 0) for metric in perf_metrics]
        
        if any(perf_values):
            # Normalize values for visualization
            perf_values_norm = [v / max(perf_values) if max(perf_values) > 0 else 0 for v in perf_values]
            bars = ax6.bar(perf_metrics, perf_values_norm, color='lightcoral', alpha=0.7)
            ax6.set_title('Processing Performance (Normalized)')
            ax6.set_ylabel('Normalized Value')
            ax6.tick_params(axis='x', rotation=45)
            
            # Add actual value labels
            for bar, actual_value in zip(bars, perf_values):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{actual_value:.2f}', ha='center', va='bottom')
        else:
            ax6.text(0.5, 0.5, 'Performance data not available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Processing Performance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Depth statistics plot saved to {save_path}")
        
        return fig
    
    def plot_gaussian_statistics(self, gaussian_stats: Dict, save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create comprehensive Gaussian reconstruction statistics visualization.
        
        Args:
            gaussian_stats: Gaussian statistics dictionary
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('4D Gaussian Reconstruction Statistics', fontsize=16, fontweight='bold')
        
        # 1. Gaussian count distribution
        ax1 = axes[0, 0]
        if 'frame_gaussian_counts' in gaussian_stats:
            counts = gaussian_stats['frame_gaussian_counts']
            ax1.hist(counts, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.set_title('Gaussian Count Distribution')
            ax1.set_xlabel('Gaussians per Frame')
            ax1.set_ylabel('Frequency')
            ax1.axvline(np.mean(counts), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(counts):.0f}')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Gaussian count data not available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Gaussian Count Distribution')
        
        # 2. Temporal evolution of Gaussian counts
        ax2 = axes[0, 1]
        if 'frame_gaussian_counts' in gaussian_stats:
            counts = gaussian_stats['frame_gaussian_counts']
            frames = range(len(counts))
            
            ax2.plot(frames, counts, 'b-', alpha=0.7, linewidth=1)
            ax2.fill_between(frames, counts, alpha=0.3)
            ax2.set_title('Gaussian Count Evolution')
            ax2.set_xlabel('Frame Number')
            ax2.set_ylabel('Number of Gaussians')
            
            # Add trend line
            if len(counts) > 1:
                z = np.polyfit(frames, counts, 1)
                p = np.poly1d(z)
                ax2.plot(frames, p(frames), "r--", alpha=0.8, 
                        label=f'Trend: {z[0]:.2f}x + {z[1]:.0f}')
                ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Temporal data not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Gaussian Count Evolution')
        
        # 3. Spatial distribution
        ax3 = axes[0, 2]
        if 'spatial_bounds' in gaussian_stats:
            bounds = gaussian_stats['spatial_bounds']
            ranges = [
                bounds['max_x'] - bounds['min_x'],
                bounds['max_y'] - bounds['min_y'], 
                bounds['max_z'] - bounds['min_z']
            ]
            labels = ['X Range', 'Y Range', 'Z Range']
            colors = ['red', 'green', 'blue']
            
            bars = ax3.bar(labels, ranges, color=colors, alpha=0.7)
            ax3.set_title('Spatial Coverage')
            ax3.set_ylabel('Range')
            
            # Add value labels
            for bar, value in zip(bars, ranges):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ranges)*0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'Spatial bounds not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Spatial Coverage')
        
        # 4. Quality metrics
        ax4 = axes[1, 0]
        quality_metrics = self.analyze_gaussian_quality(gaussian_stats)
        
        if 'gaussian_consistency' in quality_metrics:
            consistency = quality_metrics['gaussian_consistency']
            metrics = ['Mean Count', 'Std Count', 'Min Count', 'Max Count']
            values = [
                consistency['mean_count'],
                consistency['std_count'],
                consistency['min_count'],
                consistency['max_count']
            ]
            
            bars = ax4.bar(metrics, values, color='lightgreen', alpha=0.7)
            ax4.set_title('Gaussian Count Statistics')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:.0f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'Consistency metrics not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Gaussian Count Statistics')
        
        # 5. Processing efficiency
        ax5 = axes[1, 1]
        if 'processing_time' in gaussian_stats and 'total_frames' in gaussian_stats:
            total_time = gaussian_stats['processing_time']
            total_frames = gaussian_stats['total_frames']
            time_per_frame = total_time / total_frames if total_frames > 0 else 0
            
            metrics = ['Total Time (s)', 'Time per Frame (s)', 'Frames per Second']
            values = [total_time, time_per_frame, 1/time_per_frame if time_per_frame > 0 else 0]
            
            bars = ax5.bar(metrics, values, color='orange', alpha=0.7)
            ax5.set_title('Processing Efficiency')
            ax5.set_ylabel('Time / Rate')
            ax5.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        else:
            ax5.text(0.5, 0.5, 'Processing time not available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Processing Efficiency')
        
        # 6. Memory usage (if available)
        ax6 = axes[1, 2]
        if 'memory_usage' in gaussian_stats:
            memory_stats = gaussian_stats['memory_usage']
            stages = list(memory_stats.keys())
            usage_mb = [memory_stats[stage] / (1024*1024) for stage in stages]  # Convert to MB
            
            bars = ax6.bar(stages, usage_mb, color='purple', alpha=0.7)
            ax6.set_title('Memory Usage by Stage')
            ax6.set_ylabel('Memory (MB)')
            ax6.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, usage_mb):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(usage_mb)*0.01,
                        f'{value:.1f}', ha='center', va='bottom')
        else:
            ax6.text(0.5, 0.5, 'Memory usage not available', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Memory Usage')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gaussian statistics plot saved to {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, save_path: Optional[Union[str, Path]] = None) -> str:
        """
        Create a comprehensive statistics report covering all pipeline stages.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report text
        """
        report_lines = [
            "COMPREHENSIVE PIPELINE STATISTICS REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Output Directory: {self.output_dir}",
            ""
        ]
        
        # Depth statistics
        depth_stats = self.load_depth_statistics()
        if depth_stats:
            report_lines.extend([
                "DEPTH ESTIMATION STATISTICS",
                "-" * 35,
                f"Mean Depth: {depth_stats.get('mean_depth', 'N/A')}",
                f"Depth Standard Deviation: {depth_stats.get('std_depth', 'N/A')}",
                f"Depth Range: {depth_stats.get('min_depth', 'N/A')} - {depth_stats.get('max_depth', 'N/A')}",
                f"Coverage: {depth_stats.get('coverage_percentage', 'N/A')}%",
                ""
            ])
            
            # Quality analysis
            quality_metrics = self.analyze_depth_quality(depth_stats)
            if quality_metrics:
                report_lines.extend([
                    "Depth Quality Analysis:",
                    f"  Depth Consistency: {quality_metrics.get('depth_consistency', 'N/A'):.3f}",
                    f"  Depth Range: {quality_metrics.get('depth_range', 'N/A'):.2f}",
                    ""
                ])
        else:
            report_lines.extend([
                "DEPTH ESTIMATION STATISTICS",
                "-" * 35,
                "No depth statistics available",
                ""
            ])
        
        # Gaussian statistics
        gaussian_stats = self.load_gaussian_statistics()
        if gaussian_stats:
            report_lines.extend([
                "4D GAUSSIAN RECONSTRUCTION STATISTICS", 
                "-" * 45,
                f"Total Gaussians: {gaussian_stats.get('total_gaussians', 'N/A'):,}",
                f"Total Frames: {gaussian_stats.get('total_frames', 'N/A')}",
                ""
            ])
            
            # Frame statistics
            if 'frame_gaussian_counts' in gaussian_stats:
                counts = gaussian_stats['frame_gaussian_counts']
                report_lines.extend([
                    "Per-Frame Statistics:",
                    f"  Average Gaussians per Frame: {np.mean(counts):.0f}",
                    f"  Min Gaussians: {np.min(counts)}",
                    f"  Max Gaussians: {np.max(counts)}",
                    f"  Standard Deviation: {np.std(counts):.2f}",
                    ""
                ])
            
            # Quality analysis
            quality_metrics = self.analyze_gaussian_quality(gaussian_stats)
            if 'gaussian_consistency' in quality_metrics:
                consistency = quality_metrics['gaussian_consistency']
                report_lines.extend([
                    "Gaussian Quality Analysis:",
                    f"  Coefficient of Variation: {consistency['coefficient_of_variation']:.3f}",
                    ""
                ])
            
            # Spatial coverage
            if 'spatial_coverage' in quality_metrics:
                coverage = quality_metrics['spatial_coverage']
                report_lines.extend([
                    "Spatial Coverage:",
                    f"  X Range: {coverage['x_range']:.2f}",
                    f"  Y Range: {coverage['y_range']:.2f}",
                    f"  Z Range: {coverage['z_range']:.2f}",
                    f"  Total Volume: {coverage['volume']:.2f}",
                    ""
                ])
        else:
            report_lines.extend([
                "4D GAUSSIAN RECONSTRUCTION STATISTICS",
                "-" * 45,
                "No Gaussian statistics available",
                ""
            ])
        
        # Performance summary
        report_lines.extend([
            "PERFORMANCE SUMMARY",
            "-" * 25
        ])
        
        total_processing_time = 0
        if depth_stats and 'processing_time' in depth_stats:
            total_processing_time += depth_stats['processing_time']
            report_lines.append(f"Depth Processing Time: {depth_stats['processing_time']:.2f}s")
        
        if gaussian_stats and 'processing_time' in gaussian_stats:
            total_processing_time += gaussian_stats['processing_time']
            report_lines.append(f"Gaussian Processing Time: {gaussian_stats['processing_time']:.2f}s")
        
        if total_processing_time > 0:
            report_lines.append(f"Total Processing Time: {total_processing_time:.2f}s")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Comprehensive report saved to {save_path}")
        
        return report_text
    
    def create_interactive_dashboard(self) -> Optional[go.Figure]:
        """
        Create an interactive statistics dashboard using Plotly.
        
        Returns:
            Plotly figure or None if Plotly not available
        """
        if not HAS_PLOTLY:
            logging.warning("Plotly not available for interactive dashboard")
            return None
        
        # Load data
        depth_stats = self.load_depth_statistics()
        gaussian_stats = self.load_gaussian_statistics()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Depth Metrics', 'Gaussian Count Evolution',
                           'Quality Metrics', 'Processing Performance',
                           'Coverage Statistics', 'Memory Usage'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # 1. Depth metrics
        if depth_stats:
            metrics = ['mean_depth', 'std_depth', 'min_depth', 'max_depth']
            values = [depth_stats.get(metric, 0) for metric in metrics]
            
            fig.add_trace(
                go.Bar(x=metrics, y=values, name="Depth Metrics"),
                row=1, col=1
            )
        
        # 2. Gaussian evolution
        if gaussian_stats and 'frame_gaussian_counts' in gaussian_stats:
            counts = gaussian_stats['frame_gaussian_counts']
            frames = list(range(len(counts)))
            
            fig.add_trace(
                go.Scatter(x=frames, y=counts, mode='lines+markers', 
                          name="Gaussian Count"),
                row=1, col=2
            )
        
        # 3. Quality metrics
        quality_data = []
        if depth_stats:
            depth_quality = self.analyze_depth_quality(depth_stats)
            for key, value in depth_quality.items():
                if isinstance(value, (int, float)):
                    quality_data.append((f"Depth: {key}", value))
        
        if gaussian_stats:
            gaussian_quality = self.analyze_gaussian_quality(gaussian_stats)
            for key, value in gaussian_quality.items():
                if isinstance(value, (int, float)):
                    quality_data.append((f"Gaussian: {key}", value))
        
        if quality_data:
            labels, values = zip(*quality_data)
            fig.add_trace(
                go.Bar(x=list(labels), y=list(values), name="Quality Metrics"),
                row=2, col=1
            )
        
        # 4. Processing performance
        perf_data = []
        if depth_stats and 'processing_time' in depth_stats:
            perf_data.append(('Depth Processing', depth_stats['processing_time']))
        if gaussian_stats and 'processing_time' in gaussian_stats:
            perf_data.append(('Gaussian Processing', gaussian_stats['processing_time']))
        
        if perf_data:
            labels, times = zip(*perf_data)
            fig.add_trace(
                go.Bar(x=list(labels), y=list(times), name="Processing Time"),
                row=2, col=2
            )
        
        # 5. Coverage pie chart
        if depth_stats and 'coverage_percentage' in depth_stats:
            coverage = depth_stats['coverage_percentage']
            fig.add_trace(
                go.Pie(labels=['Covered', 'Not Covered'], 
                      values=[coverage, 100-coverage], name="Coverage"),
                row=3, col=1
            )
        
        # 6. Memory usage (placeholder)
        fig.add_trace(
            go.Bar(x=['Depth', 'Gaussian'], y=[100, 200], name="Memory (MB)"),
            row=3, col=2
        )
        
        fig.update_layout(
            title="Pipeline Statistics Dashboard",
            height=900,
            showlegend=False
        )
        
        return fig
    
    def compare_quality_across_datasets(self, dataset_patterns: List[str]) -> plt.Figure:
        """
        Compare quality metrics across multiple datasets.
        
        Args:
            dataset_patterns: List of dataset name patterns to compare
            
        Returns:
            Matplotlib figure with comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Quality Comparison Across Datasets', fontsize=16, fontweight='bold')
        
        datasets_data = []
        dataset_names = []
        
        for pattern in dataset_patterns:
            # Load depth stats
            depth_stats = self.load_depth_statistics(pattern)
            gaussian_stats = self.load_gaussian_statistics(pattern)
            
            if depth_stats or gaussian_stats:
                datasets_data.append({
                    'depth': depth_stats,
                    'gaussian': gaussian_stats,
                    'name': pattern
                })
                dataset_names.append(pattern)
        
        if not datasets_data:
            # No data found
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No data available for comparison', 
                       ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # 1. Depth quality comparison
        ax1 = axes[0, 0]
        depth_metrics = []
        for data in datasets_data:
            if data['depth']:
                depth_quality = self.analyze_depth_quality(data['depth'])
                depth_metrics.append(depth_quality.get('depth_consistency', 0))
            else:
                depth_metrics.append(0)
        
        bars = ax1.bar(dataset_names, depth_metrics, color='skyblue', alpha=0.7)
        ax1.set_title('Depth Consistency Comparison')
        ax1.set_ylabel('Consistency Metric')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Gaussian count comparison
        ax2 = axes[0, 1]
        gaussian_counts = []
        for data in datasets_data:
            if data['gaussian'] and 'total_gaussians' in data['gaussian']:
                gaussian_counts.append(data['gaussian']['total_gaussians'])
            else:
                gaussian_counts.append(0)
        
        bars = ax2.bar(dataset_names, gaussian_counts, color='lightgreen', alpha=0.7)
        ax2.set_title('Total Gaussians Comparison')
        ax2.set_ylabel('Number of Gaussians')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Processing time comparison
        ax3 = axes[1, 0]
        processing_times = []
        for data in datasets_data:
            total_time = 0
            if data['depth'] and 'processing_time' in data['depth']:
                total_time += data['depth']['processing_time']
            if data['gaussian'] and 'processing_time' in data['gaussian']:
                total_time += data['gaussian']['processing_time']
            processing_times.append(total_time)
        
        bars = ax3.bar(dataset_names, processing_times, color='lightcoral', alpha=0.7)
        ax3.set_title('Total Processing Time Comparison')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Summary table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        headers = ['Dataset', 'Depth Quality', 'Gaussians', 'Time (s)']
        
        for i, data in enumerate(datasets_data):
            row = [
                dataset_names[i],
                f"{depth_metrics[i]:.3f}" if depth_metrics[i] > 0 else "N/A",
                f"{gaussian_counts[i]:,}" if gaussian_counts[i] > 0 else "N/A",
                f"{processing_times[i]:.1f}" if processing_times[i] > 0 else "N/A"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Comparison')
        
        plt.tight_layout()
        return fig
