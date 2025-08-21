"""
Base class for all trajectory quality assessment metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np


class BaseMetric(ABC):
    """
    Abstract base class for trajectory quality assessment metrics.
    
    All assessment metrics should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the metric.
        
        Args:
            name: Name of the metric
            description: Description of what the metric measures
        """
        self.name = name
        self.description = description
        self.results = {}
        self.is_calculated = False
    
    @abstractmethod
    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the metric value.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            Dictionary containing the calculated metric results
        """
        pass
    
    @abstractmethod
    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """
        Validate input data for the metric calculation.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the calculated results.
        
        Returns:
            Dictionary containing the metric results
        """
        if not self.is_calculated:
            raise RuntimeError(f"Metric {self.name} has not been calculated yet.")
        return self.results.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the metric results.
        
        Returns:
            Dictionary containing summary information
        """
        if not self.is_calculated:
            return {"name": self.name, "status": "not_calculated"}
        
        return {
            "name": self.name,
            "description": self.description,
            "status": "calculated",
            "results_keys": list(self.results.keys())
        }
    
    def reset(self):
        """Reset the metric to initial state."""
        self.results = {}
        self.is_calculated = False 