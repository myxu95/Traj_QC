"""
Basic tests for Traj_QC package.
"""

import pytest
import numpy as np
import tempfile
import os
import sys

# Add the package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from traj_qc.core.base_metric import BaseMetric
from traj_qc.config.manager import ConfigManager
from traj_qc.metrics.structural_metrics import RMSDMetric, RMSFMetric, RadiusOfGyrationMetric


class TestBaseMetric:
    """Test the base metric class."""
    
    def test_base_metric_initialization(self):
        """Test base metric initialization."""
        metric = RMSDMetric()
        assert metric.name == "rmsd"
        assert metric.description == "Root Mean Square Deviation from reference structure"
        assert not metric.is_calculated
        assert metric.results == {}
    
    def test_base_metric_reset(self):
        """Test metric reset functionality."""
        metric = RMSDMetric()
        metric.results = {"test": "data"}
        metric.is_calculated = True
        
        metric.reset()
        assert metric.results == {}
        assert not metric.is_calculated


class TestConfigManager:
    """Test the configuration manager."""
    
    def test_default_config_loading(self):
        """Test default configuration loading."""
        config_manager = ConfigManager()
        
        # Check that default config is loaded
        assert "general" in config_manager.config
        assert "metrics" in config_manager.config
        
        # Check some default values
        assert config_manager.get_output_dir() == "trajectory_quality_output"
        assert "rmsd" in config_manager.get_metric_configs()
    
    def test_metric_enabling_disabling(self):
        """Test enabling and disabling metrics."""
        config_manager = ConfigManager()
        
        # Initially RMSD should be enabled
        assert config_manager.is_metric_enabled("rmsd")
        
        # Disable RMSD
        config_manager.disable_metric("rmsd")
        assert not config_manager.is_metric_enabled("rmsd")
        
        # Enable RMSD again
        config_manager.enable_metric("rmsd")
        assert config_manager.is_metric_enabled("rmsd")


class TestStructuralMetrics:
    """Test structural analysis metrics."""
    
    def test_rmsd_metric_initialization(self):
        """Test RMSD metric initialization."""
        metric = RMSDMetric(reference_frame=5, selection="protein")
        assert metric.reference_frame == 5
        assert metric.selection == "protein"
    
    def test_rmsf_metric_initialization(self):
        """Test RMSF metric initialization."""
        metric = RMSFMetric(selection="backbone")
        assert metric.selection == "backbone"
    
    def test_rg_metric_initialization(self):
        """Test radius of gyration metric initialization."""
        metric = RadiusOfGyrationMetric(selection="protein")
        assert metric.selection == "protein"
    
    def test_metric_validation(self):
        """Test metric input validation."""
        metric = RMSDMetric()
        
        # Valid data
        valid_data = {
            'n_frames': 100,
            'coordinates': np.random.random((100, 10, 3)),
            'reader': 'MDAnalysis'
        }
        assert metric.validate_input(valid_data)
        
        # Invalid data (missing required keys)
        invalid_data = {'n_frames': 100}
        assert not metric.validate_input(invalid_data)


class TestIntegration:
    """Test integration between components."""
    
    def test_config_with_metrics(self):
        """Test configuration manager with metric configurations."""
        config_manager = ConfigManager()
        
        # Get metric configs
        metric_configs = config_manager.get_metric_configs()
        
        # Check that RMSD config exists and has expected structure
        assert "rmsd" in metric_configs
        rmsd_config = metric_configs["rmsd"]
        assert "enabled" in rmsd_config
        assert "parameters" in rmsd_config
        assert "reference_frame" in rmsd_config["parameters"]
    
    def test_metric_parameter_access(self):
        """Test accessing metric parameters from config."""
        config_manager = ConfigManager()
        
        rmsd_config = config_manager.get_metric_config("rmsd")
        assert rmsd_config is not None
        assert rmsd_config["parameters"]["reference_frame"] == 0
        assert rmsd_config["parameters"]["selection"] == "protein and name CA"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 