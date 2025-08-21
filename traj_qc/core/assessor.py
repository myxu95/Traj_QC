"""
Main trajectory quality assessment orchestrator.
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from .base_metric import BaseMetric
from ..io.trajectory_reader import TrajectoryReader
from ..config.manager import ConfigManager


class TrajectoryAssessor:
    """
    Main class for coordinating trajectory quality assessment.
    
    This class manages the execution of various assessment metrics
    and generates comprehensive quality reports.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trajectory assessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_manager = ConfigManager(config_path) if config_path else ConfigManager()
        self.trajectory_reader = TrajectoryReader()
        self.metrics: Dict[str, BaseMetric] = {}
        self.trajectory_data: Dict[str, Any] = {}
        self.assessment_results: Dict[str, Any] = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load configured metrics
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics based on configuration."""
        metric_configs = self.config_manager.get_metric_configs()
        
        for metric_name, config in metric_configs.items():
            if config.get("enabled", True):
                metric_class = self._get_metric_class(metric_name)
                if metric_class:
                    self.metrics[metric_name] = metric_class(**config.get("parameters", {}))
                    self.logger.info(f"Loaded metric: {metric_name}")
    
    def _get_metric_class(self, metric_name: str):
        """Get metric class by name."""
        # This will be implemented to dynamically import metric classes
        # For now, return None as placeholder
        return None
    
    def load_trajectory(self, trajectory_path: str, topology_path: str = None):
        """
        Load trajectory data for analysis.
        
        Args:
            trajectory_path: Path to trajectory file
            topology_path: Path to topology file (optional)
        """
        self.logger.info(f"Loading trajectory from: {trajectory_path}")
        
        try:
            self.trajectory_data = self.trajectory_reader.read_trajectory(
                trajectory_path, topology_path
            )
            self.logger.info("Trajectory loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load trajectory: {e}")
            raise
    
    def run_assessment(self) -> Dict[str, Any]:
        """
        Run all enabled assessment metrics.
        
        Returns:
            Dictionary containing all assessment results
        """
        if not self.trajectory_data:
            raise RuntimeError("No trajectory data loaded. Call load_trajectory() first.")
        
        self.logger.info("Starting trajectory quality assessment...")
        
        for metric_name, metric in self.metrics.items():
            try:
                self.logger.info(f"Calculating metric: {metric_name}")
                
                if metric.validate_input(self.trajectory_data):
                    results = metric.calculate(self.trajectory_data)
                    self.assessment_results[metric_name] = results
                    self.logger.info(f"Metric {metric_name} calculated successfully")
                else:
                    self.logger.warning(f"Metric {metric_name} input validation failed")
                    self.assessment_results[metric_name] = {"error": "Input validation failed"}
                    
            except Exception as e:
                self.logger.error(f"Error calculating metric {metric_name}: {e}")
                self.assessment_results[metric_name] = {"error": str(e)}
        
        self.logger.info("Trajectory quality assessment completed")
        return self.assessment_results.copy()
    
    def get_metric_results(self, metric_name: str) -> Dict[str, Any]:
        """
        Get results for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary containing metric results
        """
        if metric_name not in self.assessment_results:
            raise KeyError(f"Metric {metric_name} not found in assessment results")
        
        return self.assessment_results[metric_name]
    
    def get_all_results(self) -> Dict[str, Any]:
        """
        Get all assessment results.
        
        Returns:
            Dictionary containing all metric results
        """
        return self.assessment_results.copy()
    
    def generate_report(self, output_path: str = "trajectory_quality_report.html"):
        """
        Generate a comprehensive quality assessment report.
        
        Args:
            output_path: Path for the output report file
        """
        if not self.assessment_results:
            raise RuntimeError("No assessment results available. Run assessment first.")
        
        self.logger.info(f"Generating report to: {output_path}")
        
        # This will be implemented to generate HTML/PDF reports
        # For now, just log the action
        self.logger.info("Report generation not yet implemented")
    
    def add_custom_metric(self, metric: BaseMetric):
        """
        Add a custom metric to the assessor.
        
        Args:
            metric: Custom metric instance inheriting from BaseMetric
        """
        if not isinstance(metric, BaseMetric):
            raise TypeError("Metric must inherit from BaseMetric")
        
        self.metrics[metric.name] = metric
        self.logger.info(f"Added custom metric: {metric.name}")
    
    def remove_metric(self, metric_name: str):
        """
        Remove a metric from the assessor.
        
        Args:
            metric_name: Name of the metric to remove
        """
        if metric_name in self.metrics:
            del self.metrics[metric_name]
            self.logger.info(f"Removed metric: {metric_name}")
        
        if metric_name in self.assessment_results:
            del self.assessment_results[metric_name] 