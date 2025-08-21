"""
Physical rationality analysis metrics for trajectory quality assessment.

This module provides metrics for validating the physical reasonableness
of molecular dynamics trajectories.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..core.base_metric import BaseMetric


class AtomicDisplacementMetric(BaseMetric):
    """
    Atomic displacement analysis metric.

    Analyzes atomic displacements to detect unphysical movements
    and assess trajectory quality.
    """

    def __init__(self, selection: str = "protein", 
                 max_displacement: float = 10.0,
                 displacement_window: int = 10):
        """
        Initialize atomic displacement metric.

        Args:
            selection: Atom selection string
            max_displacement: Maximum allowed displacement in Angstroms
            displacement_window: Window size for displacement analysis
        """
        super().__init__(
            name="atomic_displacement",
            description="Analysis of atomic displacements for physical validation"
        )
        self.selection = selection
        self.max_displacement = max_displacement
        self.displacement_window = displacement_window

    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """
        Validate input data for displacement analysis.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            True if input is valid, False otherwise
        """
        required_keys = ['n_frames', 'reader']
        return all(key in trajectory_data for key in required_keys)

    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate atomic displacement metrics.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            Dictionary containing displacement analysis results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for displacement analysis")

        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']

        # Calculate displacements for all frames
        displacements = self._calculate_displacements(trajectory_data, reader)
        
        # Analyze displacement patterns
        displacement_analysis = self._analyze_displacements(displacements)
        
        # Detect unphysical movements
        unphysical_movements = self._detect_unphysical_movements(displacements)
        
        # Calculate displacement quality score
        quality_score = self._calculate_displacement_quality(displacements, displacement_analysis)

        self.results = {
            'displacements': displacements,
            'selection': self.selection,
            'max_displacement': self.max_displacement,
            'displacement_window': self.displacement_window,
            'displacement_analysis': displacement_analysis,
            'unphysical_movements': unphysical_movements,
            'quality_score': quality_score,
            'n_frames': n_frames
        }

        self.is_calculated = True
        return self.results

    def _calculate_displacements(self, trajectory_data: Dict[str, Any], reader: str) -> np.ndarray:
        """Calculate atomic displacements between consecutive frames."""
        n_frames = trajectory_data['n_frames']
        displacements = []

        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            selection = universe.select_atoms(self.selection)
            n_atoms = len(selection)

            # Get first frame coordinates
            universe.trajectory[0]
            prev_coords = selection.positions.copy()

            for frame_idx in range(1, n_frames):
                universe.trajectory[frame_idx]
                current_coords = selection.positions
                
                # Calculate displacement
                frame_displacement = np.linalg.norm(current_coords - prev_coords, axis=1)
                displacements.append(frame_displacement)
                
                prev_coords = current_coords.copy()

        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            selection = trajectory.topology.select(self.selection)
            
            if selection is None or len(selection) == 0:
                raise ValueError(f"No atoms found for selection: {self.selection}")

            n_atoms = len(selection)

            for frame_idx in range(1, n_frames):
                prev_coords = trajectory.xyz[frame_idx-1, selection]
                current_coords = trajectory.xyz[frame_idx, selection]
                
                # Calculate displacement
                frame_displacement = np.linalg.norm(current_coords - prev_coords, axis=1)
                displacements.append(frame_displacement)

        return np.array(displacements)

    def _analyze_displacements(self, displacements: np.ndarray) -> Dict[str, Any]:
        """Analyze displacement patterns and statistics."""
        if len(displacements) == 0:
            return {
                'mean_displacement': 0.0,
                'std_displacement': 0.0,
                'max_displacement': 0.0,
                'displacement_distribution': {},
                'frame_displacements': []
            }

        # Calculate per-frame statistics
        frame_displacements = []
        for frame_disp in displacements:
            frame_displacements.append({
                'mean': float(np.mean(frame_disp)),
                'std': float(np.std(frame_disp)),
                'max': float(np.max(frame_disp)),
                'min': float(np.min(frame_disp))
            })

        # Calculate overall statistics
        all_displacements = displacements.flatten()
        mean_displacement = float(np.mean(all_displacements))
        std_displacement = float(np.std(all_displacements))
        max_displacement = float(np.max(all_displacements))
        min_displacement = float(np.min(all_displacements))

        # Create displacement distribution
        hist, bins = np.histogram(all_displacements, bins=20, range=(0, max_displacement))
        displacement_distribution = {
            'bins': bins.tolist(),
            'counts': hist.tolist(),
            'bin_centers': [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
        }

        return {
            'mean_displacement': mean_displacement,
            'std_displacement': std_displacement,
            'max_displacement': max_displacement,
            'min_displacement': min_displacement,
            'displacement_distribution': displacement_distribution,
            'frame_displacements': frame_displacements
        }

    def _detect_unphysical_movements(self, displacements: np.ndarray) -> Dict[str, Any]:
        """Detect potentially unphysical atomic movements."""
        if len(displacements) == 0:
            return {
                'unphysical_frames': [],
                'unphysical_atoms': [],
                'total_violations': 0
            }

        unphysical_frames = []
        unphysical_atoms = []
        total_violations = 0

        # Check for frames with excessive displacements
        for frame_idx, frame_disp in enumerate(displacements):
            excessive_disps = frame_disp > self.max_displacement
            if np.any(excessive_disps):
                unphysical_frames.append({
                    'frame': frame_idx + 1,  # +1 because we start from frame 1
                    'n_violations': int(np.sum(excessive_disps)),
                    'max_displacement': float(np.max(frame_disp)),
                    'mean_displacement': float(np.mean(frame_disp))
                })
                total_violations += int(np.sum(excessive_disps))

        # Check for atoms with consistently high displacements
        atom_displacements = np.mean(displacements, axis=0)
        problematic_atoms = np.where(atom_displacements > self.max_displacement * 0.5)[0]
        
        for atom_idx in problematic_atoms:
            unphysical_atoms.append({
                'atom_index': int(atom_idx),
                'mean_displacement': float(atom_displacements[atom_idx]),
                'max_displacement': float(np.max(displacements[:, atom_idx]))
            })

        return {
            'unphysical_frames': unphysical_frames,
            'unphysical_atoms': unphysical_atoms,
            'total_violations': total_violations,
            'violation_threshold': self.max_displacement
        }

    def _calculate_displacement_quality(self, displacements: np.ndarray, 
                                     displacement_analysis: Dict[str, Any]) -> float:
        """Calculate overall displacement quality score (0-1)."""
        if len(displacements) == 0:
            return 1.0

        # Base score
        base_score = 0.5
        
        # Penalty for excessive displacements
        max_disp = displacement_analysis['max_displacement']
        if max_disp > self.max_displacement:
            penalty = min(0.4, (max_disp - self.max_displacement) / self.max_displacement * 0.4)
            base_score -= penalty
        
        # Bonus for low average displacement
        mean_disp = displacement_analysis['mean_displacement']
        if mean_disp < self.max_displacement * 0.1:
            bonus = 0.3
        elif mean_disp < self.max_displacement * 0.3:
            bonus = 0.2
        elif mean_disp < self.max_displacement * 0.5:
            bonus = 0.1
        else:
            bonus = 0.0
        
        base_score += bonus
        
        # Penalty for violations
        total_violations = displacement_analysis['unphysical_movements']['total_violations']
        total_atoms = displacements.shape[1] if len(displacements) > 0 else 1
        violation_ratio = total_violations / (len(displacements) * total_atoms)
        
        if violation_ratio > 0.1:
            penalty = min(0.2, violation_ratio * 2)
            base_score -= penalty

        return max(0.0, min(1.0, base_score))


class BondGeometryMetric(BaseMetric):
    """
    Bond geometry validation metric.

    Monitors bond lengths and angles to ensure they remain
    within physically reasonable ranges.
    """

    def __init__(self, bond_length_tolerance: float = 0.1,
                 bond_angle_tolerance: float = 5.0,
                 selection: str = "protein"):
        """
        Initialize bond geometry metric.

        Args:
            bond_length_tolerance: Tolerance for bond length deviations in Angstroms
            bond_angle_tolerance: Tolerance for bond angle deviations in degrees
            selection: Atom selection string
        """
        super().__init__(
            name="bond_geometry",
            description="Bond length and angle validation for physical reasonableness"
        )
        self.bond_length_tolerance = bond_length_tolerance
        self.bond_angle_tolerance = bond_angle_tolerance
        self.selection = selection

    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """
        Validate input data for bond geometry analysis.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            True if input is valid, False otherwise
        """
        required_keys = ['n_frames', 'reader']
        return all(key in trajectory_data for key in required_keys)

    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate bond geometry metrics.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            Dictionary containing bond geometry analysis results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for bond geometry analysis")

        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']

        # Analyze bond lengths
        bond_length_analysis = self._analyze_bond_lengths(trajectory_data, reader)
        
        # Analyze bond angles
        bond_angle_analysis = self._analyze_bond_angles(trajectory_data, reader)
        
        # Detect geometry violations
        geometry_violations = self._detect_geometry_violations(
            bond_length_analysis, bond_angle_analysis
        )
        
        # Calculate geometry quality score
        quality_score = self._calculate_geometry_quality(
            bond_length_analysis, bond_angle_analysis, geometry_violations
        )

        self.results = {
            'bond_length_tolerance': self.bond_length_tolerance,
            'bond_angle_tolerance': self.bond_angle_tolerance,
            'selection': self.selection,
            'bond_length_analysis': bond_length_analysis,
            'bond_angle_analysis': bond_angle_analysis,
            'geometry_violations': geometry_violations,
            'quality_score': quality_score,
            'n_frames': n_frames
        }

        self.is_calculated = True
        return self.results

    def _analyze_bond_lengths(self, trajectory_data: Dict[str, Any], reader: str) -> Dict[str, Any]:
        """Analyze bond lengths throughout the trajectory."""
        n_frames = trajectory_data['n_frames']
        bond_lengths = []

        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            selection = universe.select_atoms(self.selection)
            
            # Get bonds from topology
            bonds = self._get_bonds_mdanalysis(universe, selection)
            
            for frame_idx in range(n_frames):
                universe.trajectory[frame_idx]
                frame_bond_lengths = self._calculate_frame_bond_lengths(
                    universe, bonds, selection
                )
                bond_lengths.append(frame_bond_lengths)

        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            selection = trajectory.topology.select(self.selection)
            
            if selection is None or len(selection) == 0:
                raise ValueError(f"No atoms found for selection: {self.selection}")

            # Get bonds from topology
            bonds = self._get_bonds_mdtraj(trajectory, selection)
            
            for frame_idx in range(n_frames):
                frame_bond_lengths = self._calculate_frame_bond_lengths_mdtraj(
                    trajectory, bonds, selection, frame_idx
                )
                bond_lengths.append(frame_bond_lengths)

        bond_lengths = np.array(bond_lengths)
        
        # Calculate statistics
        mean_bond_lengths = np.mean(bond_lengths, axis=0)
        std_bond_lengths = np.std(bond_lengths, axis=0)
        
        return {
            'bond_lengths': bond_lengths.tolist(),
            'mean_bond_lengths': mean_bond_lengths.tolist(),
            'std_bond_lengths': std_bond_lengths.tolist(),
            'n_bonds': len(mean_bond_lengths),
            'bonds': bonds
        }

    def _analyze_bond_angles(self, trajectory_data: Dict[str, Any], reader: str) -> Dict[str, Any]:
        """Analyze bond angles throughout the trajectory."""
        n_frames = trajectory_data['n_frames']
        bond_angles = []

        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            selection = universe.select_atoms(self.selection)
            
            # Get angles from topology
            angles = self._get_angles_mdanalysis(universe, selection)
            
            for frame_idx in range(n_frames):
                universe.trajectory[frame_idx]
                frame_bond_angles = self._calculate_frame_bond_angles(
                    universe, angles, selection
                )
                bond_angles.append(frame_bond_angles)

        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            selection = trajectory.topology.select(self.selection)
            
            if selection is None or len(selection) == 0:
                raise ValueError(f"No atoms found for selection: {self.selection}")

            # Get angles from topology
            angles = self._get_angles_mdtraj(trajectory, selection)
            
            for frame_idx in range(n_frames):
                frame_bond_angles = self._calculate_frame_bond_angles_mdtraj(
                    trajectory, angles, selection, frame_idx
                )
                bond_angles.append(frame_bond_angles)

        bond_angles = np.array(bond_angles)
        
        # Calculate statistics
        mean_bond_angles = np.mean(bond_angles, axis=0)
        std_bond_angles = np.std(bond_angles, axis=0)
        
        return {
            'bond_angles': bond_angles.tolist(),
            'mean_bond_angles': mean_bond_angles.tolist(),
            'std_bond_angles': std_bond_angles.tolist(),
            'n_angles': len(mean_bond_angles),
            'angles': angles
        }

    def _detect_geometry_violations(self, bond_length_analysis: Dict[str, Any],
                                  bond_angle_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect bond geometry violations."""
        violations = {
            'bond_length_violations': [],
            'bond_angle_violations': [],
            'total_violations': 0
        }

        # Check bond length violations
        mean_bond_lengths = np.array(bond_length_analysis['mean_bond_lengths'])
        std_bond_lengths = np.array(bond_length_analysis['std_bond_lengths'])
        
        for i, (mean_len, std_len) in enumerate(zip(mean_bond_lengths, std_bond_lengths)):
            if std_len > self.bond_length_tolerance:
                violations['bond_length_violations'].append({
                    'bond_index': i,
                    'mean_length': float(mean_len),
                    'std_length': float(std_len),
                    'tolerance': self.bond_length_tolerance,
                    'violation_type': 'high_variability'
                })
                violations['total_violations'] += 1

        # Check bond angle violations
        mean_bond_angles = np.array(bond_angle_analysis['mean_bond_angles'])
        std_bond_angles = np.array(bond_angle_analysis['std_bond_angles'])
        
        for i, (mean_angle, std_angle) in enumerate(zip(mean_bond_angles, std_bond_angles)):
            if std_angle > self.bond_angle_tolerance:
                violations['bond_angle_violations'].append({
                    'angle_index': i,
                    'mean_angle': float(mean_angle),
                    'std_angle': float(std_angle),
                    'tolerance': self.bond_angle_tolerance,
                    'violation_type': 'high_variability'
                })
                violations['total_violations'] += 1

        return violations

    def _calculate_geometry_quality(self, bond_length_analysis: Dict[str, Any],
                                  bond_angle_analysis: Dict[str, Any],
                                  geometry_violations: Dict[str, Any]) -> float:
        """Calculate overall geometry quality score (0-1)."""
        # Base score
        base_score = 0.7
        
        # Penalty for violations
        total_violations = geometry_violations['total_violations']
        total_geometries = (bond_length_analysis['n_bonds'] + 
                           bond_angle_analysis['n_angles'])
        
        if total_geometries > 0:
            violation_ratio = total_violations / total_geometries
            penalty = min(0.3, violation_ratio * 3)
            base_score -= penalty
        
        # Bonus for low variability
        if bond_length_analysis['n_bonds'] > 0:
            avg_bond_std = np.mean(bond_length_analysis['std_bond_lengths'])
            if avg_bond_std < self.bond_length_tolerance * 0.5:
                base_score += 0.1
        
        if bond_angle_analysis['n_angles'] > 0:
            avg_angle_std = np.mean(bond_angle_analysis['std_bond_angles'])
            if avg_angle_std < self.bond_angle_tolerance * 0.5:
                base_score += 0.1

        return max(0.0, min(1.0, base_score))

    def _get_bonds_mdanalysis(self, universe, selection):
        """Get bonds from MDAnalysis universe."""
        try:
            # Try to get bonds from topology
            bonds = selection.bonds
            return [(bond.atoms[0].index, bond.atoms[1].index) for bond in bonds]
        except:
            # Fallback: create bonds between consecutive atoms
            indices = selection.indices
            bonds = []
            for i in range(len(indices) - 1):
                bonds.append((indices[i], indices[i + 1]))
            return bonds

    def _get_bonds_mdtraj(self, trajectory, selection):
        """Get bonds from MDTraj topology."""
        try:
            bonds = []
            for bond in trajectory.topology.bonds:
                if bond.atom1.index in selection and bond.atom2.index in selection:
                    bonds.append((bond.atom1.index, bond.atom2.index))
            return bonds
        except:
            # Fallback: create bonds between consecutive atoms
            bonds = []
            for i in range(len(selection) - 1):
                bonds.append((selection[i], selection[i + 1]))
            return bonds

    def _get_angles_mdanalysis(self, universe, selection):
        """Get angles from MDAnalysis universe."""
        try:
            # Create angles from bonds
            bonds = self._get_bonds_mdanalysis(universe, selection)
            angles = []
            for i in range(len(bonds) - 1):
                if bonds[i][1] == bonds[i + 1][0]:
                    angles.append((bonds[i][0], bonds[i][1], bonds[i + 1][1]))
            return angles
        except:
            return []

    def _get_angles_mdtraj(self, trajectory, selection):
        """Get angles from MDTraj topology."""
        try:
            angles = []
            for angle in trajectory.topology.angles:
                if (angle.atom1.index in selection and 
                    angle.atom2.index in selection and 
                    angle.atom3.index in selection):
                    angles.append((angle.atom1.index, angle.atom2.index, angle.atom3.index))
            return angles
        except:
            return []

    def _calculate_frame_bond_lengths(self, universe, bonds, selection):
        """Calculate bond lengths for a single frame using MDAnalysis."""
        bond_lengths = []
        for bond in bonds:
            atom1_pos = universe.atoms[bond[0]].position
            atom2_pos = universe.atoms[bond[1]].position
            length = np.linalg.norm(atom1_pos - atom2_pos)
            bond_lengths.append(length)
        return np.array(bond_lengths)

    def _calculate_frame_bond_lengths_mdtraj(self, trajectory, bonds, selection, frame_idx):
        """Calculate bond lengths for a single frame using MDTraj."""
        bond_lengths = []
        for bond in bonds:
            atom1_pos = trajectory.xyz[frame_idx, bond[0]]
            atom2_pos = trajectory.xyz[frame_idx, bond[1]]
            length = np.linalg.norm(atom1_pos - atom2_pos)
            bond_lengths.append(length)
        return np.array(bond_lengths)

    def _calculate_frame_bond_angles(self, universe, angles, selection):
        """Calculate bond angles for a single frame using MDAnalysis."""
        bond_angles = []
        for angle in angles:
            atom1_pos = universe.atoms[angle[0]].position
            atom2_pos = universe.atoms[angle[1]].position
            atom3_pos = universe.atoms[angle[2]].position
            
            # Calculate angle
            v1 = atom1_pos - atom2_pos
            v2 = atom3_pos - atom2_pos
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            bond_angles.append(angle_deg)
        return np.array(bond_angles)

    def _calculate_frame_bond_angles_mdtraj(self, trajectory, angles, selection, frame_idx):
        """Calculate bond angles for a single frame using MDTraj."""
        bond_angles = []
        for angle in angles:
            atom1_pos = trajectory.xyz[frame_idx, angle[0]]
            atom2_pos = trajectory.xyz[frame_idx, angle[1]]
            atom3_pos = trajectory.xyz[frame_idx, angle[2]]
            
            # Calculate angle
            v1 = atom1_pos - atom2_pos
            v2 = atom3_pos - atom2_pos
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            bond_angles.append(angle_deg)
        return np.array(bond_angles) 