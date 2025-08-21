"""
Plotting utilities for trajectory quality assessment.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_quality_plots(assessment_results: Dict[str, Any], 
                        output_dir: str = "plots",
                        plot_format: str = "png") -> Dict[str, str]:
    """
    Create comprehensive quality assessment plots.
    
    Args:
        assessment_results: Dictionary containing assessment results
        output_dir: Directory to save plots
        plot_format: Plot file format (png, pdf, html)
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    plot_files = {}
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create structural stability plots
    if 'rmsd' in assessment_results:
        plot_files['rmsd'] = _create_rmsd_plot(
            assessment_results['rmsd'], output_dir, plot_format
        )
    
    if 'rmsf' in assessment_results:
        plot_files['rmsf'] = _create_rmsf_plot(
            assessment_results['rmsf'], output_dir, plot_format
        )
    
    if 'radius_of_gyration' in assessment_results:
        plot_files['rg'] = _create_rg_plot(
            assessment_results['radius_of_gyration'], output_dir, plot_format
        )
    
    # Create hydrogen bond plots
    if 'hydrogen_bonds' in assessment_results:
        plot_files['hbonds'] = _create_hbond_plot(
            assessment_results['hydrogen_bonds'], output_dir, plot_format
        )
    
    # Create secondary structure plots
    if 'secondary_structure' in assessment_results:
        plot_files['ss'] = _create_ss_plot(
            assessment_results['secondary_structure'], output_dir, plot_format
        )
    
    # Create energy plots
    if 'energy_analysis' in assessment_results:
        plot_files['energy'] = _create_energy_plot(
            assessment_results['energy_analysis'], output_dir, plot_format
        )
    
    # Create summary dashboard
    plot_files['dashboard'] = _create_summary_dashboard(
        assessment_results, output_dir, plot_format
    )
    
    return plot_files


def _create_rmsd_plot(rmsd_results: Dict[str, Any], output_dir: str, plot_format: str) -> str:
    """Create RMSD plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    times = np.arange(len(rmsd_results['rmsd_values']))
    rmsd_values = rmsd_results['rmsd_values']
    
    # Time series plot
    ax1.plot(times, rmsd_values, 'b-', linewidth=1, alpha=0.7)
    ax1.axhline(y=rmsd_results['mean_rmsd'], color='r', linestyle='--', 
                label=f'Mean: {rmsd_results["mean_rmsd"]:.3f} Å')
    ax1.fill_between(times, 
                     rmsd_results['mean_rmsd'] - rmsd_results['std_rmsd'],
                     rmsd_results['mean_rmsd'] + rmsd_results['std_rmsd'],
                     alpha=0.3, color='r')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('RMSD (Å)')
    ax1.set_title('RMSD Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution plot
    ax2.hist(rmsd_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=rmsd_results['mean_rmsd'], color='r', linestyle='--', 
                label=f'Mean: {rmsd_results["mean_rmsd"]:.3f} Å')
    ax2.set_xlabel('RMSD (Å)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('RMSD Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"rmsd_analysis.{plot_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def _create_rmsf_plot(rmsf_results: Dict[str, Any], output_dir: str, plot_format: str) -> str:
    """Create RMSF plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rmsf_values = rmsf_results['rmsf_values']
    residue_indices = np.arange(len(rmsf_values))
    
    ax.plot(residue_indices, rmsf_values, 'b-', linewidth=1, alpha=0.8)
    ax.fill_between(residue_indices, rmsf_values, alpha=0.3, color='skyblue')
    
    # Highlight high flexibility regions
    threshold = rmsf_results['mean_rmsf'] + rmsf_results['std_rmsf']
    high_flex = rmsf_values > threshold
    ax.scatter(residue_indices[high_flex], rmsf_values[high_flex], 
               color='red', s=20, alpha=0.7, label='High flexibility')
    
    ax.axhline(y=rmsf_results['mean_rmsf'], color='r', linestyle='--', 
               label=f'Mean: {rmsf_results["mean_rmsf"]:.3f} Å')
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('RMSF (Å)')
    ax.set_title('Root Mean Square Fluctuation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"rmsf_analysis.{plot_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def _create_rg_plot(rg_results: Dict[str, Any], output_dir: str, plot_format: str) -> str:
    """Create radius of gyration plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    times = np.arange(len(rg_results['rg_values']))
    rg_values = rg_results['rg_values']
    
    # Time series plot
    ax1.plot(times, rg_values, 'g-', linewidth=1, alpha=0.7)
    ax1.axhline(y=rg_results['mean_rg'], color='r', linestyle='--', 
                label=f'Mean: {rg_results["mean_rg"]:.3f} Å')
    ax1.fill_between(times, 
                     rg_results['mean_rg'] - rg_results['std_rg'],
                     rg_results['mean_rg'] + rg_results['std_rg'],
                     alpha=0.3, color='r')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Radius of Gyration (Å)')
    ax1.set_title('Radius of Gyration Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution plot
    ax2.hist(rg_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(x=rg_results['mean_rg'], color='r', linestyle='--', 
                label=f'Mean: {rg_results["mean_rg"]:.3f} Å')
    ax2.set_xlabel('Radius of Gyration (Å)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Radius of Gyration Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"rg_analysis.{plot_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def _create_hbond_plot(hbond_results: Dict[str, Any], output_dir: str, plot_format: str) -> str:
    """Create hydrogen bond plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    times = np.arange(len(hbond_results['hbond_counts']))
    hbond_counts = hbond_results['hbond_counts']
    
    # Time series plot
    ax1.plot(times, hbond_counts, 'purple', linewidth=1, alpha=0.7)
    ax1.axhline(y=hbond_results['mean_hbonds'], color='r', linestyle='--', 
                label=f'Mean: {hbond_results["mean_hbonds"]:.1f}')
    ax1.fill_between(times, 
                     hbond_results['mean_hbonds'] - hbond_results['std_hbonds'],
                     hbond_results['mean_hbonds'] + hbond_results['std_hbonds'],
                     alpha=0.3, color='r')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Number of Hydrogen Bonds')
    ax1.set_title('Hydrogen Bond Count Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution plot
    ax2.hist(hbond_counts, bins=30, alpha=0.7, color='plum', edgecolor='black')
    ax2.axvline(x=hbond_results['mean_hbonds'], color='r', linestyle='--', 
                label=f'Mean: {hbond_results["mean_hbonds"]:.1f}')
    ax2.set_xlabel('Number of Hydrogen Bonds')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Hydrogen Bond Count Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"hbond_analysis.{plot_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def _create_ss_plot(ss_results: Dict[str, Any], output_dir: str, plot_format: str) -> str:
    """Create secondary structure plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ss_counts = ss_results['ss_counts']
    n_frames = ss_results['n_frames']
    times = np.arange(n_frames)
    
    # Time series of secondary structure content
    for ss_type, counts in ss_counts.items():
        if len(counts) > 0:
            ax1.plot(times, counts, label=ss_type, alpha=0.8, linewidth=1)
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Count')
    ax1.set_title('Secondary Structure Content Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar plot of average content
    ss_means = [ss_results['ss_statistics'][ss_type]['mean'] for ss_type in ss_results['ss_types']]
    ss_stds = [ss_results['ss_statistics'][ss_type]['std'] for ss_type in ss_results['ss_types']]
    
    bars = ax2.bar(ss_results['ss_types'], ss_means, yerr=ss_stds, 
                   alpha=0.7, capsize=5)
    ax2.set_xlabel('Secondary Structure Type')
    ax2.set_ylabel('Average Count')
    ax2.set_title('Average Secondary Structure Content')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, ss_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"ss_analysis.{plot_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def _create_energy_plot(energy_results: Dict[str, Any], output_dir: str, plot_format: str) -> str:
    """Create energy plot."""
    if 'error' in energy_results:
        return ""
    
    energy_data = energy_results['energy_data']
    times = energy_data['Time']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each energy component
    for i, col_name in enumerate(energy_results['energy_columns']):
        if col_name in energy_data and i < 4:
            ax = axes[i]
            energies = energy_data[col_name]
            
            ax.plot(times, energies, alpha=0.7, linewidth=1)
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('Energy (kJ/mol)')
            ax.set_title(f'{col_name} Energy')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats = energy_results['energy_statistics'][col_name]
            ax.axhline(y=stats['mean'], color='r', linestyle='--', 
                      label=f'Mean: {stats["mean"]:.1f}')
    
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"energy_analysis.{plot_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def _create_summary_dashboard(assessment_results: Dict[str, Any], output_dir: str, plot_format: str) -> str:
    """Create summary dashboard with key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Collect summary statistics
    summary_data = []
    
    for i, (metric_name, results) in enumerate(assessment_results.items()):
        if 'error' in results:
            continue
            
        ax = axes[i] if i < 6 else axes[-1]
        
        if metric_name == 'rmsd':
            ax.text(0.5, 0.5, f'RMSD\nMean: {results["mean_rmsd"]:.3f} Å\nStd: {results["std_rmsd"]:.3f} Å',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax.set_title('RMSD Summary')
            
        elif metric_name == 'rmsf':
            ax.text(0.5, 0.5, f'RMSF\nMean: {results["mean_rmsf"]:.3f} Å\nMax: {results["max_rmsf"]:.3f} Å',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            ax.set_title('RMSF Summary')
            
        elif metric_name == 'radius_of_gyration':
            ax.text(0.5, 0.5, f'Radius of Gyration\nMean: {results["mean_rg"]:.3f} Å\nStd: {results["std_rg"]:.3f} Å',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax.set_title('Rg Summary')
            
        elif metric_name == 'hydrogen_bonds':
            ax.text(0.5, 0.5, f'Hydrogen Bonds\nMean: {results["mean_hbonds"]:.1f}\nStd: {results["std_hbonds"]:.1f}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.7))
            ax.set_title('H-Bonds Summary')
            
        elif metric_name == 'secondary_structure':
            stability = results['ss_stability']['stability_score']
            ax.text(0.5, 0.5, f'Secondary Structure\nStability: {stability:.3f}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
            ax.set_title('SS Summary')
            
        elif metric_name == 'energy_analysis':
            if 'energy_conservation' in results:
                conservation = results['energy_conservation']['conservation_score']
                ax.text(0.5, 0.5, f'Energy Conservation\nScore: {conservation:.3f}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue", alpha=0.7))
                ax.set_title('Energy Summary')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(len(assessment_results), 6):
        axes[i].set_visible(False)
    
    plt.suptitle('Trajectory Quality Assessment Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"quality_dashboard.{plot_format}")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename 