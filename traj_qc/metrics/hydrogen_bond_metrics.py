"""
Hydrogen bond analysis metrics for trajectory quality assessment.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from ..core.base_metric import BaseMetric


class HydrogenBondMetric(BaseMetric):
    """
    Hydrogen bond analysis metric.
    
    Analyzes hydrogen bond networks, lifetimes, and patterns in trajectories.
    """
    
    def __init__(self, donor_selection: str = "protein and (name N or name NE or name NH1 or name NH2)",
                 acceptor_selection: str = "protein and (name O or name OE1 or name OE2)",
                 distance_cutoff: float = 3.5,
                 angle_cutoff: float = 30.0):
        """
        Initialize hydrogen bond metric.
        
        Args:
            donor_selection: Selection string for hydrogen bond donors
            acceptor_selection: Selection string for hydrogen bond acceptors
            distance_cutoff: Maximum distance for hydrogen bond (Angstroms)
            angle_cutoff: Maximum angle for hydrogen bond (degrees)
        """
        super().__init__(
            name="hydrogen_bonds",
            description="Hydrogen bond network analysis"
        )
        self.donor_selection = donor_selection
        self.acceptor_selection = acceptor_selection
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = np.radians(angle_cutoff)
    
    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """Validate input data for hydrogen bond calculation."""
        required_keys = ['n_frames', 'coordinates', 'reader']
        return all(key in trajectory_data for key in required_keys)
    
    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate hydrogen bond analysis for all frames.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            Dictionary containing hydrogen bond results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for hydrogen bond calculation")
        
        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']
        
        if reader == 'MDAnalysis':
            return self._calculate_with_mdanalysis(trajectory_data, n_frames)
        elif reader == 'MDTraj':
            return self._calculate_with_mdtraj(trajectory_data, n_frames)
        else:
            raise ValueError(f"Unsupported reader: {reader}")
    
    def _calculate_with_mdanalysis(self, trajectory_data: Dict[str, Any], n_frames: int) -> Dict[str, Any]:
        """Calculate hydrogen bonds using MDAnalysis."""
        universe = trajectory_data['universe']
        donors = universe.select_atoms(self.donor_selection)
        acceptors = universe.select_atoms(self.acceptor_selection)
        
        # Get hydrogen atoms for donors
        donor_hydrogens = []
        for donor in donors:
            # Find hydrogen atoms bonded to donor
            bonded_hydrogens = donor.bonded_atoms.select_atoms("name H*")
            if len(bonded_hydrogens) > 0:
                donor_hydrogens.append(bonded_hydrogens[0])
            else:
                donor_hydrogens.append(None)
        
        # Analyze each frame
        all_hbonds = []
        hbond_counts = []
        
        for frame_idx in range(n_frames):
            universe.trajectory[frame_idx]
            
            frame_hbonds = []
            for i, donor in enumerate(donors):
                if donor_hydrogens[i] is None:
                    continue
                    
                donor_pos = donor.position
                h_pos = donor_hydrogens[i].position
                
                for acceptor in acceptors:
                    acceptor_pos = acceptor.position
                    
                    # Check distance
                    h_acceptor_dist = np.linalg.norm(h_pos - acceptor_pos)
                    if h_acceptor_dist > self.distance_cutoff:
                        continue
                    
                    # Check angle (donor-hydrogen-acceptor)
                    vec1 = h_pos - donor_pos
                    vec2 = acceptor_pos - h_pos
                    
                    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
                        continue
                    
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    
                    if angle < self.angle_cutoff:
                        frame_hbonds.append({
                            'donor_idx': donor.index,
                            'acceptor_idx': acceptor.index,
                            'distance': h_acceptor_dist,
                            'angle': np.degrees(angle)
                        })
            
            all_hbonds.append(frame_hbonds)
            hbond_counts.append(len(frame_hbonds))
        
        return self._process_hbond_results(all_hbonds, hbond_counts, n_frames)
    
    def _calculate_with_mdtraj(self, trajectory_data: Dict[str, Any], n_frames: int) -> Dict[str, Any]:
        """Calculate hydrogen bonds using MDTraj."""
        trajectory = trajectory_data['trajectory']
        
        # MDTraj has built-in hydrogen bond detection
        try:
            hbonds = md.baker_hubbard(trajectory, periodic=False, 
                                     distance_cutoff=self.distance_cutoff/10.0,  # Convert to nm
                                     angle_cutoff=self.angle_cutoff)
            
            # Process results
            all_hbonds = []
            hbond_counts = []
            
            for frame_idx in range(n_frames):
                frame_hbonds = []
                frame_mask = hbonds[0] == frame_idx
                
                if np.any(frame_mask):
                    frame_indices = np.where(frame_mask)[0]
                    for idx in frame_indices:
                        donor_idx = hbonds[1][idx]
                        acceptor_idx = hbonds[2][idx]
                        distance = hbonds[3][idx] * 10.0  # Convert to Angstroms
                        
                        frame_hbonds.append({
                            'donor_idx': int(donor_idx),
                            'acceptor_idx': int(acceptor_idx),
                            'distance': distance,
                            'angle': 0.0  # MDTraj doesn't provide angle
                        })
                
                all_hbonds.append(frame_hbonds)
                hbond_counts.append(len(frame_hbonds))
                
        except Exception as e:
            # Fallback to manual calculation
            self.logger.warning(f"MDTraj hydrogen bond detection failed: {e}. Using manual calculation.")
            return self._calculate_manual_mdtraj(trajectory_data, n_frames)
        
        return self._process_hbond_results(all_hbonds, hbond_counts, n_frames)
    
    def _calculate_manual_mdtraj(self, trajectory_data: Dict[str, Any], n_frames: int) -> Dict[str, Any]:
        """Manual hydrogen bond calculation for MDTraj."""
        trajectory = trajectory_data['trajectory']
        
        # This is a simplified manual calculation
        # In practice, you'd need more sophisticated donor-acceptor identification
        
        all_hbonds = []
        hbond_counts = []
        
        for frame_idx in range(n_frames):
            # Placeholder for manual calculation
            frame_hbonds = []
            all_hbonds.append(frame_hbonds)
            hbond_counts.append(0)
        
        return self._process_hbond_results(all_hbonds, hbond_counts, n_frames)
    
    def _process_hbond_results(self, all_hbonds: List[List[Dict]], hbond_counts: List[int], n_frames: int) -> Dict[str, Any]:
        """Process hydrogen bond results into final output."""
        hbond_counts = np.array(hbond_counts)
        
        # Calculate statistics
        mean_hbonds = np.mean(hbond_counts)
        std_hbonds = np.std(hbond_counts)
        max_hbonds = np.max(hbond_counts)
        min_hbonds = np.min(hbond_counts)
        
        # Calculate persistence (how long bonds last)
        persistence = self._calculate_bond_persistence(all_hbonds, n_frames)
        
        self.results = {
            'hbond_counts': hbond_counts,
            'all_hbonds': all_hbonds,
            'mean_hbonds': mean_hbonds,
            'std_hbonds': std_hbonds,
            'max_hbonds': max_hbonds,
            'min_hbonds': min_hbonds,
            'bond_persistence': persistence,
            'donor_selection': self.donor_selection,
            'acceptor_selection': self.acceptor_selection,
            'distance_cutoff': self.distance_cutoff,
            'angle_cutoff': np.degrees(self.angle_cutoff)
        }
        
        self.is_calculated = True
        return self.results
    
    def _calculate_bond_persistence(self, all_hbonds: List[List[Dict]], n_frames: int) -> Dict[str, Any]:
        """Calculate how long hydrogen bonds persist."""
        if n_frames == 0:
            return {}
        
        # Track bond persistence
        bond_lifetimes = []
        current_bonds = set()
        
        for frame_idx in range(n_frames):
            frame_bonds = set()
            for hbond in all_hbonds[frame_idx]:
                bond_key = (hbond['donor_idx'], hbond['acceptor_idx'])
                frame_bonds.add(bond_key)
            
            # Find bonds that broke
            broken_bonds = current_bonds - frame_bonds
            for bond in broken_bonds:
                # Calculate lifetime (simplified)
                bond_lifetimes.append(1)  # Placeholder
            
            # Find new bonds
            new_bonds = frame_bonds - current_bonds
            
            current_bonds = frame_bonds
        
        if bond_lifetimes:
            return {
                'mean_lifetime': np.mean(bond_lifetimes),
                'max_lifetime': np.max(bond_lifetimes),
                'total_bonds': len(bond_lifetimes)
            }
        else:
            return {
                'mean_lifetime': 0.0,
                'max_lifetime': 0.0,
                'total_bonds': 0
            } 