"""
Trajectory quality assessment metrics organized by analysis categories.
"""

# Structural stability metrics
from .structural_stability import RMSDMetric, RMSFMetric

# Energy analysis metrics
from .energy_analysis import TotalEnergyMetric, TemperaturePressureMetric

# Binding state analysis metrics
from .binding_analysis import ContactAreaMetric, ContactEnergyMetric

# Convergence analysis metrics
from .convergence_analysis import RMSDConvergenceMetric, TrajectoryClusteringMetric

# Physical rationality metrics
from .physical_rationality import AtomicDisplacementMetric, BondGeometryMetric

# Binding region analysis metrics
from .binding_region_analysis import BindingRegionRMSFMetric, BindingRegionContactMetric

# Legacy metrics (for backward compatibility)
from .hydrogen_bond_metrics import HydrogenBondMetric
from .secondary_structure_metrics import SecondaryStructureMetric

__all__ = [
    # Structural stability
    "RMSDMetric",
    "RMSFMetric",

    # Energy analysis
    "TotalEnergyMetric",
    "TemperaturePressureMetric",

    # Binding state analysis
    "ContactAreaMetric",
    "ContactEnergyMetric",

    # Convergence analysis
    "RMSDConvergenceMetric",
    "TrajectoryClusteringMetric",

    # Physical rationality
    "AtomicDisplacementMetric",
    "BondGeometryMetric",

    # Binding region analysis
    "BindingRegionRMSFMetric",
    "BindingRegionContactMetric",

    # Legacy metrics
    "HydrogenBondMetric",
    "SecondaryStructureMetric"
] 