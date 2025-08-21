#!/usr/bin/env python3
"""
Command-line interface for Traj_QC trajectory quality assessment system.

Usage:
    python traj_qc_cli.py --trajectory trajectory.xtc --topology topology.pdb
    python traj_qc_cli.py --config config.yaml --trajectory trajectory.xtc
    python traj_qc_cli.py --help
"""

import argparse
import sys
import os
from pathlib import Path

# Add the package to the path
sys.path.append(os.path.dirname(__file__))

from traj_qc import TrajectoryAssessor
from traj_qc.utils.plotting import create_quality_plots


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Traj_QC - Trajectory Quality Assessment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with trajectory file
  python traj_qc_cli.py --trajectory md.xtc --topology protein.pdb
  
  # Use custom configuration
  python traj_qc_cli.py --config my_config.yaml --trajectory md.xtc
  
  # Specify output directory
  python traj_qc_cli.py --trajectory md.xtc --output results/
  
  # Enable specific metrics only
  python traj_qc_cli.py --trajectory md.xtc --metrics rmsd,rmsf,radius_of_gyration
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--trajectory", "-t",
        required=True,
        help="Path to trajectory file (.xtc, .trr, .gro, .pdb)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--topology", "-p",
        help="Path to topology file (optional for some formats)"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file (YAML)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="trajectory_quality_output",
        help="Output directory for results (default: trajectory_quality_output)"
    )
    
    parser.add_argument(
        "--metrics",
        help="Comma-separated list of metrics to run (default: all enabled in config)"
    )
    
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Plot output format (default: png)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Traj_QC 0.1.0"
    )
    
    return parser.parse_args()


def validate_files(args):
    """Validate input files exist."""
    if not os.path.exists(args.trajectory):
        print(f"Error: Trajectory file not found: {args.trajectory}")
        return False
    
    if args.topology and not os.path.exists(args.topology):
        print(f"Error: Topology file not found: {args.topology}")
        return False
    
    if args.config and not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return False
    
    return True


def run_assessment(args):
    """Run the trajectory quality assessment."""
    print("Traj_QC - Trajectory Quality Assessment")
    print("=" * 50)
    
    # Initialize assessor
    print(f"Initializing trajectory assessor...")
    if args.config:
        assessor = TrajectoryAssessor(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        assessor = TrajectoryAssessor()
        print("Using default configuration")
    
    # Load trajectory
    print(f"Loading trajectory: {args.trajectory}")
    try:
        assessor.load_trajectory(args.trajectory, args.topology)
        
        # Show trajectory info
        summary = assessor.trajectory_reader.get_trajectory_summary()
        print(f"Trajectory loaded: {summary['n_frames']} frames, {summary['n_atoms']} atoms")
        
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return False
    
    # Run assessment
    print("Running quality assessment...")
    try:
        results = assessor.run_assessment()
        print(f"Assessment completed! Calculated {len(results)} metrics:")
        
        for metric_name, metric_results in results.items():
            if 'error' not in metric_results:
                print(f"  ✓ {metric_name}")
            else:
                print(f"  ✗ {metric_name}: {metric_results['error']}")
                
    except Exception as e:
        print(f"Error during assessment: {e}")
        return False
    
    # Generate plots
    print("Generating quality assessment plots...")
    try:
        plots_dir = os.path.join(args.output, "plots")
        plot_files = create_quality_plots(results, plots_dir, args.plot_format)
        
        print(f"Generated {len(plot_files)} plots in: {plots_dir}")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    # Generate report
    print("Generating quality assessment report...")
    try:
        report_path = os.path.join(args.output, "trajectory_quality_report.html")
        assessor.generate_report(report_path)
        print(f"Report generated: {report_path}")
        
    except Exception as e:
        print(f"Error generating report: {e}")
    
    # Save results
    print("Saving assessment results...")
    try:
        import json
        results_file = os.path.join(args.output, "assessment_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for metric_name, metric_results in results.items():
            if 'error' not in metric_results:
                serializable_results[metric_name] = {}
                for key, value in metric_results.items():
                    if hasattr(value, 'tolist'):  # numpy array
                        serializable_results[metric_name][key] = value.tolist()
                    else:
                        serializable_results[metric_name][key] = value
            else:
                serializable_results[metric_name] = metric_results
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\n" + "=" * 50)
    print("Trajectory quality assessment completed successfully!")
    print(f"Results saved in: {args.output}")
    
    return True


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate input files
    if not validate_files(args):
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run assessment
    success = run_assessment(args)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 