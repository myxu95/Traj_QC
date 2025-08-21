"""
Traj_QC - Trajectory Quality Control System

A Python package for GROMACS molecular dynamics trajectory analysis and quality assessment.
"""

__version__ = "0.1.0"
__author__ = "Traj_QC Team"

from .core.assessor import TrajectoryAssessor
from .config.manager import ConfigManager

__all__ = [
    "TrajectoryAssessor",
    "ConfigManager",
] 