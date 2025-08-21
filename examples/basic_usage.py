#!/usr/bin/env python3
"""
Basic usage example for Traj_QC trajectory quality assessment system.

This example demonstrates how to:
1. Load a trajectory
2. Configure assessment metrics
3. Run quality assessment
4. Generate plots and reports
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from traj_qc import TrajectoryAssessor, ConfigManager


def main():
    """Main function demonstrating basic usage."""
    
    print("Traj_QC - Trajectory Quality Assessment Example")
    print("=" * 50)
    
    # 1. Initialize the assessor with configuration
    print("\n1. Initializing trajectory assessor...")
    
    # You can specify a custom config file or use defaults
    config_file = "config/trajectory_quality_config.yaml"
    
    if os.path.exists(config_file):
        assessor = TrajectoryAssessor(config_file)
        print(f"   Loaded configuration from: {config_file}")
    else:
        assessor = TrajectoryAssessor()
        print("   Using default configuration")
    
    # 2. Load trajectory data
    print("\n2. Loading trajectory data...")
    
    # Example trajectory file paths (modify these for your data)
    trajectory_file = "example_trajectory.xtc"  # or .trr, .gro, .pdb
    topology_file = "example_topology.pdb"     # optional
    
    if os.path.exists(trajectory_file):
        try:
            assessor.load_trajectory(trajectory_file, topology_file)
            print(f"   Successfully loaded trajectory: {trajectory_file}")
            
            # Show trajectory summary
            summary = assessor.trajectory_reader.get_trajectory_summary()
            print(f"   Trajectory info: {summary['n_frames']} frames, {summary['n_atoms']} atoms")
            
        except Exception as e:
            print(f"   Error loading trajectory: {e}")
            print("   Using demo data for demonstration...")
            # For demo purposes, we'll continue with placeholder data
    else:
        print(f"   Trajectory file not found: {trajectory_file}")
        print("   Using demo data for demonstration...")
    
    # 3. Run quality assessment
    print("\n3. Running quality assessment...")
    
    try:
        results = assessor.run_assessment()
        print(f"   Assessment completed successfully!")
        print(f"   Calculated {len(results)} metrics:")
        
        for metric_name, metric_results in results.items():
            if 'error' not in metric_results:
                print(f"     ✓ {metric_name}")
            else:
                print(f"     ✗ {metric_name}: {metric_results['error']}")
                
    except Exception as e:
        print(f"   Error during assessment: {e}")
        return
    
    # 4. Display results summary
    print("\n4. Assessment Results Summary:")
    print("-" * 40)
    
    for metric_name, metric_results in results.items():
        if 'error' in metric_results:
            continue
            
        print(f"\n{metric_name.upper()}:")
        
        if metric_name == 'rmsd':
            print(f"  Mean RMSD: {metric_results['mean_rmsd']:.3f} Å")
            print(f"  Std RMSD:  {metric_results['std_rmsd']:.3f} Å")
            print(f"  Range:     {metric_results['min_rmsd']:.3f} - {metric_results['max_rmsd']:.3f} Å")
            
        elif metric_name == 'rmsf':
            print(f"  Mean RMSF: {metric_results['mean_rmsf']:.3f} Å")
            print(f"  Max RMSF:  {metric_results['max_rmsf']:.3f} Å")
            print(f"  Atoms:     {metric_results['n_atoms']}")
            
        elif metric_name == 'radius_of_gyration':
            print(f"  Mean Rg:   {metric_results['mean_rg']:.3f} Å")
            print(f"  Std Rg:    {metric_results['std_rg']:.3f} Å")
            print(f"  Range:     {metric_results['min_rg']:.3f} - {metric_results['max_rg']:.3f} Å")
            
        elif metric_name == 'hydrogen_bonds':
            print(f"  Mean H-bonds: {metric_results['mean_hbonds']:.1f}")
            print(f"  Std H-bonds:  {metric_results['std_hbonds']:.1f}")
            print(f"  Range:        {metric_results['min_hbonds']:.0f} - {metric_results['max_hbonds']:.0f}")
            
        elif metric_name == 'secondary_structure':
            stability = metric_results['ss_stability']['stability_score']
            print(f"  Stability score: {stability:.3f}")
            print(f"  SS types: {', '.join(metric_results['ss_types'])}")
            
        elif metric_name == 'energy_analysis':
            if 'energy_conservation' in metric_results:
                conservation = metric_results['energy_conservation']['conservation_score']
                print(f"  Conservation score: {conservation:.3f}")
            print(f"  Energy components: {', '.join(metric_results['energy_columns'])}")
    
    # 5. Generate plots
    print("\n5. Generating quality assessment plots...")
    
    try:
        from traj_qc.utils.plotting import create_quality_plots
        
        output_dir = "example_output/plots"
        plot_files = create_quality_plots(results, output_dir, "png")
        
        print(f"   Generated {len(plot_files)} plots in: {output_dir}")
        for plot_name, plot_path in plot_files.items():
            print(f"     ✓ {plot_name}: {os.path.basename(plot_path)}")
            
    except Exception as e:
        print(f"   Error generating plots: {e}")
    
    # 6. Generate report
    print("\n6. Generating quality assessment report...")
    
    try:
        report_path = "example_output/trajectory_quality_report.html"
        assessor.generate_report(report_path)
        print(f"   Report generated: {report_path}")
        
    except Exception as e:
        print(f"   Error generating report: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nTo use with your own data:")
    print("1. Modify the trajectory_file and topology_file paths")
    print("2. Adjust the configuration in config/trajectory_quality_config.yaml")
    print("3. Run the script again")


if __name__ == "__main__":
    main() 