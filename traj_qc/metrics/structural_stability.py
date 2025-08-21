"""
Structural stability metrics for trajectory quality assessment.

This module provides metrics for analyzing structural stability:
- RMSD (Root Mean Square Deviation)
- RMSF (Root Mean Square Fluctuation)
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.base_metric import BaseMetric


class RMSDMetric(BaseMetric):
    """
    Root Mean Square Deviation (RMSD) metric.
    
    Measures the structural deviation of a trajectory from a reference structure.
    Essential for assessing structural stability and convergence.
    """
    
    def __init__(self, reference_frame: int = 0, selection: str = "protein and name CA"):
        """
        Initialize RMSD metric.
        
        Args:
            reference_frame: Frame index to use as reference (default: 0)
            selection: Atom selection string (default: "protein and name CA")
        """
        super().__init__(
            name="rmsd",
            description="Root Mean Square Deviation from reference structure"
        )
        self.reference_frame = reference_frame
        self.selection = selection
    
    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """Validate input data for RMSD calculation."""
        required_keys = ['n_frames', 'coordinates', 'reader']
        return all(key in trajectory_data for key in required_keys)
    
    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate RMSD values for all frames.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            Dictionary containing RMSD results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for RMSD calculation")
        
        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']
        
        # Get reference coordinates
        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            selection = universe.select_atoms(self.selection)
            reference_coords = selection.positions
            
            # Calculate RMSD for each frame
            rmsd_values = []
            for frame_idx in range(n_frames):
                universe.trajectory[frame_idx]
                current_coords = selection.positions
                
                # Align current frame to reference
                aligned_coords = self._align_coordinates(current_coords, reference_coords)
                rmsd = self._calculate_rmsd(aligned_coords, reference_coords)
                rmsd_values.append(rmsd)
                
        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            selection = trajectory.topology.select(self.selection)
            
            if selection is None or len(selection) == 0:
                raise ValueError(f"No atoms found for selection: {self.selection}")
            
            # Get reference coordinates
            reference_coords = trajectory.xyz[self.reference_frame, selection]
            
            # Calculate RMSD for each frame
            rmsd_values = []
            for frame_idx in range(n_frames):
                current_coords = trajectory.xyz[frame_idx, selection]
                
                # Align current frame to reference
                aligned_coords = self._align_coordinates(current_coords, reference_coords)
                rmsd = self._calculate_rmsd(aligned_coords, reference_coords)
                rmsd_values.append(rmsd)
        else:
            raise ValueError(f"Unsupported reader: {reader}")
        
        rmsd_values = np.array(rmsd_values)
        
        # Calculate stability indicators
        stability_score = self._calculate_stability_score(rmsd_values)
        convergence_analysis = self._analyze_convergence(rmsd_values)
        
        self.results = {
            'rmsd_values': rmsd_values,
            'reference_frame': self.reference_frame,
            'selection': self.selection,
            'mean_rmsd': np.mean(rmsd_values),
            'std_rmsd': np.std(rmsd_values),
            'max_rmsd': np.max(rmsd_values),
            'min_rmsd': np.min(rmsd_values),
            'stability_score': stability_score,
            'convergence_analysis': convergence_analysis
        }
        
        self.is_calculated = True
        return self.results
    
    def _align_coordinates(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Align coordinates using Kabsch algorithm."""
        # Center coordinates
        coords1_centered = coords1 - np.mean(coords1, axis=0)
        coords2_centered = coords2 - np.mean(coords2, axis=0)
        
        # Calculate rotation matrix
        H = coords1_centered.T @ coords2_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Apply rotation
        aligned_coords = coords1_centered @ R
        return aligned_coords + np.mean(coords2, axis=0)
    
    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate RMSD between two coordinate sets."""
        diff = coords1 - coords2
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    
    def _calculate_stability_score(self, rmsd_values: np.ndarray) -> float:
        """Calculate structural stability score (lower is more stable)."""
        # Normalize by mean RMSD and penalize high variance
        mean_rmsd = np.mean(rmsd_values)
        cv = np.std(rmsd_values) / mean_rmsd if mean_rmsd > 0 else 0
        return 1.0 / (1.0 + cv)
    
    def _analyze_convergence(self, rmsd_values: np.ndarray) -> Dict[str, Any]:
        """Analyze RMSD convergence over time."""
        if len(rmsd_values) < 10:
            return {"converged": False, "reason": "Insufficient frames"}
        
        # Split trajectory into halves
        mid_point = len(rmsd_values) // 2
        first_half = rmsd_values[:mid_point]
        second_half = rmsd_values[mid_point:]
        
        # Check if second half has lower variance
        first_std = np.std(first_half)
        second_std = np.std(second_half)
        
        # Check if RMSD is decreasing or stable
        first_mean = np.mean(first_half)
        second_mean = np.mean(second_half)
        
        # Convergence criteria
        variance_decreasing = second_std < first_std
        mean_stable = abs(second_mean - first_mean) < 0.5 * first_std
        
        return {
            "converged": variance_decreasing and mean_stable,
            "first_half_std": first_std,
            "second_half_std": second_std,
            "first_half_mean": first_mean,
            "second_half_mean": second_mean,
            "variance_decreasing": variance_decreasing,
            "mean_stable": mean_stable
        }


class RMSFMetric(BaseMetric):
    """
    Root Mean Square Fluctuation (RMSF) metric.
    
    Measures the flexibility of individual atoms/residues over time.
    Critical for identifying flexible regions and binding sites.
    """
    
    def __init__(self, selection: str = "protein and name CA"):
        """
        Initialize RMSF metric.
        
        Args:
            selection: Atom selection string (default: "protein and name CA")
        """
        super().__init__(
            name="rmsf",
            description="Root Mean Square Fluctuation of individual atoms"
        )
        self.selection = selection
    
    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """Validate input data for RMSF calculation."""
        required_keys = ['n_frames', 'coordinates', 'reader']
        return all(key in trajectory_data for key in required_keys)
    
    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate RMSF values for selected atoms.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            Dictionary containing RMSF results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for RMSF calculation")
        
        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']
        
        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            selection = universe.select_atoms(self.selection)
            n_atoms = len(selection)
            
            # Collect coordinates for all frames
            all_coords = np.zeros((n_frames, n_atoms, 3))
            for frame_idx in range(n_frames):
                universe.trajectory[frame_idx]
                all_coords[frame_idx] = selection.positions
            
        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            selection = trajectory.topology.select(self.selection)
            
            if selection is None or len(selection) == 0:
                raise ValueError(f"No atoms found for selection: {self.selection}")
            
            n_atoms = len(selection)
            all_coords = trajectory.xyz[:, selection, :]
        else:
            raise ValueError(f"Unsupported reader: {reader}")
        
        # Calculate mean position
        mean_coords = np.mean(all_coords, axis=0)
        
        # Calculate RMSF
        rmsf_values = np.sqrt(np.mean(np.sum((all_coords - mean_coords)**2, axis=2), axis=0))
        
        # Identify flexible and rigid regions
        flexible_regions = self._identify_flexible_regions(rmsf_values)
        binding_site_analysis = self._analyze_binding_sites(rmsf_values)
        
        self.results = {
            'rmsf_values': rmsf_values,
            'selection': self.selection,
            'n_atoms': n_atoms,
            'mean_rmsf': np.mean(rmsf_values),
            'std_rmsf': np.std(rmsf_values),
            'max_rmsf': np.max(rmsf_values),
            'min_rmsf': np.min(rmsf_values),
            'flexible_regions': flexible_regions,
            'binding_site_analysis': binding_site_analysis
        }
        
        self.is_calculated = True
        return self.results
    
    def _identify_flexible_regions(self, rmsf_values: np.ndarray) -> Dict[str, Any]:
        """Identify flexible and rigid regions based on RMSF."""
        mean_rmsf = np.mean(rmsf_values)
        std_rmsf = np.std(rmsf_values)
        
        # Define thresholds
        rigid_threshold = mean_rmsf - 0.5 * std_rmsf
        flexible_threshold = mean_rmsf + 1.0 * std_rmsf
        
        # Identify regions
        rigid_indices = np.where(rmsf_values < rigid_threshold)[0]
        flexible_indices = np.where(rmsf_values > flexible_threshold)[0]
        moderate_indices = np.where((rmsf_values >= rigid_threshold) & 
                                   (rmsf_values <= flexible_threshold))[0]
        
        return {
            'rigid_regions': {
                'indices': rigid_indices,
                'count': len(rigid_indices),
                'mean_rmsf': np.mean(rmsf_values[rigid_indices]) if len(rigid_indices) > 0 else 0
            },
            'flexible_regions': {
                'indices': flexible_indices,
                'count': len(flexible_indices),
                'mean_rmsf': np.mean(rmsf_values[flexible_indices]) if len(flexible_indices) > 0 else 0
            },
            'moderate_regions': {
                'indices': moderate_indices,
                'count': len(moderate_indices),
                'mean_rmsf': np.mean(rmsf_values[moderate_indices]) if len(moderate_indices) > 0 else 0
            }
        }
    
    def _analyze_binding_sites(self, rmsf_values: np.ndarray) -> Dict[str, Any]:
        """Analyze potential binding sites based on RMSF patterns."""
        # High flexibility often indicates binding sites or active regions
        high_flex_threshold = np.percentile(rmsf_values, 75)
        high_flex_indices = np.where(rmsf_values > high_flex_threshold)[0]
        
        # Low flexibility often indicates structural cores
        low_flex_threshold = np.percentile(rmsf_values, 25)
        low_flex_indices = np.where(rmsf_values < low_flex_threshold)[0]
        
        return {
            'high_flexibility_sites': {
                'indices': high_flex_indices,
                'count': len(high_flex_indices),
                'potential_binding_sites': len(high_flex_indices)
            },
            'structural_cores': {
                'indices': low_flex_indices,
                'count': len(low_flex_indices)
            },
            'flexibility_distribution': {
                'percentile_25': np.percentile(rmsf_values, 25),
                'percentile_50': np.percentile(rmsf_values, 50),
                'percentile_75': np.percentile(rmsf_values, 75)
            }
        } 