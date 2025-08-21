"""
Configuration manager for trajectory quality assessment.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """
    Manages configuration for trajectory quality assessment.
    
    Handles loading and parsing of YAML configuration files,
    providing access to metric configurations and global settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        if config_path:
            self.load_config(config_path)
        else:
            self._load_default_config()
    
    def load_config(self, config_path: str):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.config_path = config_path
            self.logger.info(f"Configuration loaded from: {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration."""
        self.config = {
            "general": {
                "output_dir": "trajectory_quality_output",
                "report_format": ["html", "pdf"],
                "log_level": "INFO"
            },
            "metrics": {
                "rmsd": {
                    "enabled": True,
                    "parameters": {
                        "reference_frame": 0,
                        "selection": "protein and name CA"
                    }
                },
                "rmsf": {
                    "enabled": True,
                    "parameters": {
                        "selection": "protein and name CA"
                    }
                },
                "radius_of_gyration": {
                    "enabled": True,
                    "parameters": {
                        "selection": "protein"
                    }
                },
                "hydrogen_bonds": {
                    "enabled": False,
                    "parameters": {
                        "donor_selection": "protein and (name N or name NE or name NH1 or name NH2)",
                        "acceptor_selection": "protein and (name O or name OE1 or name OE2)"
                    }
                },
                "secondary_structure": {
                    "enabled": False,
                    "parameters": {
                        "selection": "protein"
                    }
                }
            }
        }
        self.logger.info("Default configuration loaded")
    
    def get_metric_configs(self) -> Dict[str, Any]:
        """
        Get configurations for all metrics.
        
        Returns:
            Dictionary containing metric configurations
        """
        return self.config.get("metrics", {})
    
    def get_metric_config(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Metric configuration dictionary or None if not found
        """
        return self.config.get("metrics", {}).get(metric_name)
    
    def is_metric_enabled(self, metric_name: str) -> bool:
        """
        Check if a metric is enabled.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            True if metric is enabled, False otherwise
        """
        metric_config = self.get_metric_config(metric_name)
        return metric_config.get("enabled", False) if metric_config else False
    
    def get_general_config(self) -> Dict[str, Any]:
        """
        Get general configuration settings.
        
        Returns:
            Dictionary containing general configuration
        """
        return self.config.get("general", {})
    
    def get_output_dir(self) -> str:
        """
        Get output directory from configuration.
        
        Returns:
            Output directory path
        """
        return self.config.get("general", {}).get("output_dir", "trajectory_quality_output")
    
    def get_report_formats(self) -> list:
        """
        Get report formats from configuration.
        
        Returns:
            List of report formats
        """
        return self.config.get("general", {}).get("report_format", ["html"])
    
    def update_metric_config(self, metric_name: str, config: Dict[str, Any]):
        """
        Update configuration for a specific metric.
        
        Args:
            metric_name: Name of the metric
            config: New configuration dictionary
        """
        if "metrics" not in self.config:
            self.config["metrics"] = {}
        
        self.config["metrics"][metric_name] = config
        self.logger.info(f"Updated configuration for metric: {metric_name}")
    
    def enable_metric(self, metric_name: str):
        """
        Enable a specific metric.
        
        Args:
            metric_name: Name of the metric to enable
        """
        if metric_name not in self.config.get("metrics", {}):
            self.config.setdefault("metrics", {})[metric_name] = {"enabled": True}
        else:
            self.config["metrics"][metric_name]["enabled"] = True
        
        self.logger.info(f"Enabled metric: {metric_name}")
    
    def disable_metric(self, metric_name: str):
        """
        Disable a specific metric.
        
        Args:
            metric_name: Name of the metric to disable
        """
        if metric_name in self.config.get("metrics", {}):
            self.config["metrics"][metric_name]["enabled"] = False
            self.logger.info(f"Disabled metric: {metric_name}")
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration (uses current path if None)
        """
        save_path = output_path or self.config_path
        if not save_path:
            save_path = "trajectory_quality_config.yaml"
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {save_path}: {e}")
            raise 