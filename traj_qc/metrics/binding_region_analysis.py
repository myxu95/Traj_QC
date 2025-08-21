"""
Binding region analysis metrics for trajectory quality assessment.

This module provides metrics for focused analysis of specific binding regions
and their stability characteristics.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..core.base_metric import BaseMetric


class BindingRegionRMSFMetric(BaseMetric):
    """
    Binding region RMSF analysis metric.

    Performs RMSF analysis focused on specific binding regions
    to assess local flexibility and stability.
    """

    def __init__(self, binding_site_selection: str = "protein and (resid 20:30 or resid 50:60)",
                 analysis_radius: float = 10.0,
                 reference_selection: str = "protein and name CA"):
        """
        Initialize binding region RMSF metric.

        Args:
            binding_site_selection: Selection string for binding site
            analysis_radius: Radius around binding site for analysis
            reference_selection: Reference selection for RMSF calculation
        """
        super().__init__(
            name="binding_region_rmsf",
            description="RMSF analysis focused on binding regions"
        )
        self.binding_site_selection = binding_site_selection
        self.analysis_radius = analysis_radius
        self.reference_selection = reference_selection

    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """
        Validate input data for binding region RMSF analysis.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            True if input is valid, False otherwise
        """
        required_keys = ['n_frames', 'reader']
        return all(key in trajectory_data for key in required_keys)

    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate binding region RMSF metrics.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            Dictionary containing binding region RMSF analysis results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for binding region RMSF analysis")

        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']

        # Identify binding region atoms
        binding_region_atoms = self._identify_binding_region(trajectory_data, reader)
        
        # Calculate RMSF for binding region
        binding_region_rmsf = self._calculate_binding_region_rmsf(
            trajectory_data, reader, binding_region_atoms
        )
        
        # Analyze binding site stability
        binding_site_stability = self._analyze_binding_site_stability(
            binding_region_rmsf, binding_region_atoms
        )
        
        # Compare with global RMSF
        global_comparison = self._compare_with_global_rmsf(
            trajectory_data, reader, binding_region_atoms
        )
        
        # Calculate binding region quality score
        quality_score = self._calculate_binding_region_quality(
            binding_region_rmsf, binding_site_stability, global_comparison
        )

        self.results = {
            'binding_site_selection': self.binding_site_selection,
            'analysis_radius': self.analysis_radius,
            'reference_selection': self.reference_selection,
            'binding_region_atoms': binding_region_atoms,
            'binding_region_rmsf': binding_region_rmsf,
            'binding_site_stability': binding_site_stability,
            'global_comparison': global_comparison,
            'quality_score': quality_score,
            'n_frames': n_frames
        }

        self.is_calculated = True
        return self.results

    def _identify_binding_region(self, trajectory_data: Dict[str, Any], reader: str) -> Dict[str, Any]:
        """Identify atoms in the binding region."""
        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            
            # Get binding site atoms
            binding_site_atoms = universe.select_atoms(self.binding_site_selection)
            if len(binding_site_atoms) == 0:
                raise ValueError(f"No atoms found for binding site selection: {self.binding_site_selection}")
            
            # Get reference atoms
            reference_atoms = universe.select_atoms(self.reference_selection)
            
            # Find atoms within analysis radius of binding site
            binding_region_atoms = []
            binding_site_center = np.mean(binding_site_atoms.positions, axis=0)
            
            for atom in reference_atoms:
                distance = np.linalg.norm(atom.position - binding_site_center)
                if distance <= self.analysis_radius:
                    binding_region_atoms.append({
                        'atom_index': atom.index,
                        'resid': atom.resid,
                        'resname': atom.resname,
                        'name': atom.name,
                        'distance_to_binding_site': float(distance)
                    })
            
            binding_region_atoms.sort(key=lambda x: x['distance_to_binding_site'])
            
        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            
            # Get binding site atoms
            binding_site_selection = trajectory.topology.select(self.binding_site_selection)
            if binding_site_selection is None or len(binding_site_selection) == 0:
                raise ValueError(f"No atoms found for binding site selection: {self.binding_site_selection}")
            
            # Get reference atoms
            reference_selection = trajectory.topology.select(self.reference_selection)
            if reference_selection is None or len(reference_selection) == 0:
                raise ValueError(f"No atoms found for reference selection: {self.reference_selection}")
            
            # Find atoms within analysis radius of binding site
            binding_region_atoms = []
            binding_site_center = np.mean(trajectory.xyz[0, binding_site_selection], axis=0)
            
            for atom_idx in reference_selection:
                atom = trajectory.topology.atom(atom_idx)
                distance = np.linalg.norm(trajectory.xyz[0, atom_idx] - binding_site_center)
                if distance <= self.analysis_radius:
                    binding_region_atoms.append({
                        'atom_index': atom_idx,
                        'resid': atom.residue.index,
                        'resname': atom.residue.name,
                        'name': atom.name,
                        'distance_to_binding_site': float(distance)
                    })
            
            binding_region_atoms.sort(key=lambda x: x['distance_to_binding_site'])

        return {
            'n_atoms': len(binding_region_atoms),
            'atoms': binding_region_atoms,
            'binding_site_center': binding_site_center.tolist() if 'binding_site_center' in locals() else None
        }

    def _calculate_binding_region_rmsf(self, trajectory_data: Dict[str, Any], 
                                     reader: str, binding_region_atoms: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate RMSF for atoms in the binding region."""
        n_frames = trajectory_data['n_frames']
        atom_indices = [atom['atom_index'] for atom in binding_region_atoms['atoms']]
        
        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            
            # Collect coordinates for all frames
            all_coords = np.zeros((n_frames, len(atom_indices), 3))
            for frame_idx in range(n_frames):
                universe.trajectory[frame_idx]
                for i, atom_idx in enumerate(atom_indices):
                    all_coords[frame_idx, i] = universe.atoms[atom_idx].position
                    
        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            all_coords = trajectory.xyz[:, atom_indices, :]
        
        # Calculate mean position
        mean_coords = np.mean(all_coords, axis=0)
        
        # Calculate RMSF
        rmsf_values = np.sqrt(np.mean(np.sum((all_coords - mean_coords)**2, axis=2), axis=0))
        
        # Calculate per-residue RMSF
        residue_rmsf = self._calculate_residue_rmsf(rmsf_values, binding_region_atoms)
        
        return {
            'rmsf_values': rmsf_values.tolist(),
            'mean_rmsf': float(np.mean(rmsf_values)),
            'std_rmsf': float(np.std(rmsf_values)),
            'max_rmsf': float(np.max(rmsf_values)),
            'min_rmsf': float(np.min(rmsf_values)),
            'residue_rmsf': residue_rmsf,
            'atom_coordinates': all_coords.tolist()
        }

    def _calculate_residue_rmsf(self, rmsf_values: np.ndarray, 
                               binding_region_atoms: Dict[str, Any]) -> Dict[str, float]:
        """Calculate RMSF averaged per residue."""
        residue_rmsf = {}
        
        for i, atom in enumerate(binding_region_atoms['atoms']):
            resid = atom['resid']
            if resid not in residue_rmsf:
                residue_rmsf[resid] = []
            residue_rmsf[resid].append(rmsf_values[i])
        
        # Average RMSF per residue
        avg_residue_rmsf = {}
        for resid, rmsf_list in residue_rmsf.items():
            avg_residue_rmsf[f"resid_{resid}"] = float(np.mean(rmsf_list))
        
        return avg_residue_rmsf

    def _analyze_binding_site_stability(self, binding_region_rmsf: Dict[str, Any],
                                      binding_region_atoms: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stability characteristics of the binding site."""
        rmsf_values = np.array(binding_region_rmsf['rmsf_values'])
        
        # Identify stable and flexible regions
        mean_rmsf = binding_region_rmsf['mean_rmsf']
        std_rmsf = binding_region_rmsf['std_rmsf']
        
        stable_threshold = mean_rmsf - 0.5 * std_rmsf
        flexible_threshold = mean_rmsf + 0.5 * std_rmsf
        
        stable_atoms = []
        flexible_atoms = []
        
        for i, atom in enumerate(binding_region_atoms['atoms']):
            rmsf_val = rmsf_values[i]
            if rmsf_val <= stable_threshold:
                stable_atoms.append({
                    'atom_index': atom['atom_index'],
                    'resid': atom['resid'],
                    'resname': atom['resname'],
                    'name': atom['name'],
                    'rmsf': float(rmsf_val),
                    'distance_to_binding_site': atom['distance_to_binding_site']
                })
            elif rmsf_val >= flexible_threshold:
                flexible_atoms.append({
                    'atom_index': atom['atom_index'],
                    'resid': atom['resid'],
                    'resname': atom['resname'],
                    'name': atom['name'],
                    'rmsf': float(rmsf_val),
                    'distance_to_binding_site': atom['distance_to_binding_site']
                })
        
        # Calculate stability metrics
        stability_score = 1.0 - (len(flexible_atoms) / len(rmsf_values))
        
        return {
            'stable_atoms': stable_atoms,
            'flexible_atoms': flexible_atoms,
            'n_stable_atoms': len(stable_atoms),
            'n_flexible_atoms': len(flexible_atoms),
            'stability_score': float(stability_score),
            'stable_threshold': float(stable_threshold),
            'flexible_threshold': float(flexible_threshold)
        }

    def _compare_with_global_rmsf(self, trajectory_data: Dict[str, Any], 
                                 reader: str, binding_region_atoms: Dict[str, Any]) -> Dict[str, Any]:
        """Compare binding region RMSF with global RMSF."""
        # Calculate global RMSF for reference selection
        n_frames = trajectory_data['n_frames']
        
        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            reference_atoms = universe.select_atoms(self.reference_selection)
            
            all_coords = np.zeros((n_frames, len(reference_atoms), 3))
            for frame_idx in range(n_frames):
                universe.trajectory[frame_idx]
                all_coords[frame_idx] = reference_atoms.positions
                
        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            reference_selection = trajectory.topology.select(self.reference_selection)
            if reference_selection is None or len(reference_selection) == 0:
                return {'error': 'No reference atoms found'}
            
            all_coords = trajectory.xyz[:, reference_selection, :]
        
        # Calculate global RMSF
        mean_coords = np.mean(all_coords, axis=0)
        global_rmsf_values = np.sqrt(np.mean(np.sum((all_coords - mean_coords)**2, axis=2), axis=0))
        
        # Get binding region RMSF
        binding_rmsf_values = np.array(binding_region_atoms['rmsf_values'])
        
        # Compare statistics
        global_mean = float(np.mean(global_rmsf_values))
        binding_mean = float(np.mean(binding_rmsf_values))
        
        relative_flexibility = binding_mean / global_mean if global_mean > 0 else 1.0
        
        return {
            'global_mean_rmsf': global_mean,
            'binding_mean_rmsf': binding_mean,
            'relative_flexibility': float(relative_flexibility),
            'flexibility_ratio': float(binding_mean / global_mean) if global_mean > 0 else 1.0,
            'binding_more_flexible': binding_mean > global_mean
        }

    def _calculate_binding_region_quality(self, binding_region_rmsf: Dict[str, Any],
                                        binding_site_stability: Dict[str, Any],
                                        global_comparison: Dict[str, Any]) -> float:
        """Calculate overall binding region quality score (0-1)."""
        # Base score from stability
        base_score = binding_site_stability['stability_score'] * 0.6
        
        # Bonus for low RMSF variability
        rmsf_std = binding_region_rmsf['std_rmsf']
        mean_rmsf = binding_region_rmsf['mean_rmsf']
        
        if mean_rmsf > 0:
            variability_ratio = rmsf_std / mean_rmsf
            if variability_ratio < 0.3:
                base_score += 0.2
            elif variability_ratio < 0.5:
                base_score += 0.1
        
        # Bonus for balanced flexibility
        if 'relative_flexibility' in global_comparison:
            rel_flex = global_comparison['relative_flexibility']
            if 0.8 <= rel_flex <= 1.2:  # Similar to global
                base_score += 0.2
            elif 0.6 <= rel_flex <= 1.4:  # Within reasonable range
                base_score += 0.1
        
        return max(0.0, min(1.0, base_score))


class BindingRegionContactMetric(BaseMetric):
    """
    Binding region contact analysis metric.

    Analyzes contact patterns and stability in specific binding regions
    to assess binding strength and interface quality.
    """

    def __init__(self, binding_site_selection: str = "protein and (resid 20:30 or resid 50:60)",
                 contact_cutoff: float = 5.0,
                 analysis_radius: float = 10.0,
                 min_contact_frequency: float = 0.1):
        """
        Initialize binding region contact metric.

        Args:
            binding_site_selection: Selection string for binding site
            contact_cutoff: Distance cutoff for contact definition
            analysis_radius: Radius around binding site for analysis
            min_contact_frequency: Minimum frequency for stable contacts
        """
        super().__init__(
            name="binding_region_contact",
            description="Contact analysis focused on binding regions"
        )
        self.binding_site_selection = binding_site_selection
        self.contact_cutoff = contact_cutoff
        self.analysis_radius = analysis_radius
        self.min_contact_frequency = min_contact_frequency

    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """
        Validate input data for binding region contact analysis.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            True if input is valid, False otherwise
        """
        required_keys = ['n_frames', 'reader']
        return all(key in trajectory_data for key in required_keys)

    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate binding region contact metrics.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            Dictionary containing binding region contact analysis results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for binding region contact analysis")

        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']

        # Identify binding region
        binding_region = self._identify_binding_region(trajectory_data, reader)
        
        # Analyze contacts within binding region
        contact_analysis = self._analyze_binding_region_contacts(
            trajectory_data, reader, binding_region
        )
        
        # Analyze contact stability
        contact_stability = self._analyze_contact_stability(contact_analysis)
        
        # Map binding interface
        binding_interface = self._map_binding_interface(contact_analysis, binding_region)
        
        # Calculate binding strength assessment
        binding_strength = self._assess_binding_strength(
            contact_analysis, contact_stability, binding_interface
        )
        
        # Calculate overall quality score
        quality_score = self._calculate_contact_quality(
            contact_analysis, contact_stability, binding_strength
        )

        self.results = {
            'binding_site_selection': self.binding_site_selection,
            'contact_cutoff': self.contact_cutoff,
            'analysis_radius': self.analysis_radius,
            'binding_region': binding_region,
            'contact_analysis': contact_analysis,
            'contact_stability': contact_stability,
            'binding_interface': binding_interface,
            'binding_strength': binding_strength,
            'quality_score': quality_score,
            'n_frames': n_frames
        }

        self.is_calculated = True
        return self.results

    def _identify_binding_region(self, trajectory_data: Dict[str, Any], reader: str) -> Dict[str, Any]:
        """Identify the binding region for analysis."""
        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            
            # Get binding site atoms
            binding_site_atoms = universe.select_atoms(self.binding_site_selection)
            if len(binding_site_atoms) == 0:
                raise ValueError(f"No atoms found for binding site selection: {self.binding_site_selection}")
            
            # Get all atoms within analysis radius
            binding_site_center = np.mean(binding_site_atoms.positions, axis=0)
            all_atoms = universe.atoms
            
            binding_region_atoms = []
            for atom in all_atoms:
                distance = np.linalg.norm(atom.position - binding_site_center)
                if distance <= self.analysis_radius:
                    binding_region_atoms.append({
                        'atom_index': atom.index,
                        'resid': atom.resid,
                        'resname': atom.resname,
                        'name': atom.name,
                        'distance_to_binding_site': float(distance),
                        'is_binding_site': atom.index in binding_site_atoms.indices
                    })
            
        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            
            # Get binding site atoms
            binding_site_selection = trajectory.topology.select(self.binding_site_selection)
            if binding_site_selection is None or len(binding_site_selection) == 0:
                raise ValueError(f"No atoms found for binding site selection: {self.binding_site_selection}")
            
            # Get all atoms within analysis radius
            binding_site_center = np.mean(trajectory.xyz[0, binding_site_selection], axis=0)
            
            binding_region_atoms = []
            for atom_idx in range(trajectory.n_atoms):
                atom = trajectory.topology.atom(atom_idx)
                distance = np.linalg.norm(trajectory.xyz[0, atom_idx] - binding_site_center)
                if distance <= self.analysis_radius:
                    binding_region_atoms.append({
                        'atom_index': atom_idx,
                        'resid': atom.residue.index,
                        'resname': atom.residue.name,
                        'name': atom.name,
                        'distance_to_binding_site': float(distance),
                        'is_binding_site': atom_idx in binding_site_selection
                    })

        return {
            'n_atoms': len(binding_region_atoms),
            'atoms': binding_region_atoms,
            'binding_site_center': binding_site_center.tolist() if 'binding_site_center' in locals() else None
        }

    def _analyze_binding_region_contacts(self, trajectory_data: Dict[str, Any], 
                                       reader: str, binding_region: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contacts within the binding region."""
        n_frames = trajectory_data['n_frames']
        atom_indices = [atom['atom_index'] for atom in binding_region['atoms']]
        
        # Initialize contact tracking
        contact_matrix = np.zeros((len(atom_indices), len(atom_indices)))
        contact_frequencies = {}
        
        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            
            for frame_idx in range(n_frames):
                universe.trajectory[frame_idx]
                frame_contacts = self._calculate_frame_contacts_mdanalysis(
                    universe, atom_indices, frame_idx
                )
                
                # Update contact matrix and frequencies
                for contact in frame_contacts:
                    i, j = contact['atom1_idx'], contact['atom2_idx']
                    contact_matrix[i, j] += 1
                    contact_matrix[j, i] += 1
                    
                    contact_key = f"{min(i, j)}_{max(i, j)}"
                    if contact_key not in contact_frequencies:
                        contact_frequencies[contact_key] = []
                    contact_frequencies[contact_key].append(contact['distance'])
                    
        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            
            for frame_idx in range(n_frames):
                frame_contacts = self._calculate_frame_contacts_mdtraj(
                    trajectory, atom_indices, frame_idx
                )
                
                # Update contact matrix and frequencies
                for contact in frame_contacts:
                    i, j = contact['atom1_idx'], contact['atom2_idx']
                    contact_matrix[i, j] += 1
                    contact_matrix[j, i] += 1
                    
                    contact_key = f"{min(i, j)}_{max(i, j)}"
                    if contact_key not in contact_frequencies:
                        contact_frequencies[contact_key] = []
                    contact_frequencies[contact_key].append(contact['distance'])
        
        # Calculate contact statistics
        total_contacts = np.sum(contact_matrix) / 2  # Divide by 2 for symmetric matrix
        mean_contacts_per_frame = total_contacts / n_frames
        
        # Identify stable contacts
        stable_contacts = []
        for contact_key, distances in contact_frequencies.items():
            frequency = len(distances) / n_frames
            if frequency >= self.min_contact_frequency:
                atom1_idx, atom2_idx = map(int, contact_key.split('_'))
                stable_contacts.append({
                    'atom1_idx': atom1_idx,
                    'atom2_idx': atom2_idx,
                    'frequency': float(frequency),
                    'mean_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances))
                })
        
        return {
            'contact_matrix': contact_matrix.tolist(),
            'contact_frequencies': contact_frequencies,
            'total_contacts': int(total_contacts),
            'mean_contacts_per_frame': float(mean_contacts_per_frame),
            'stable_contacts': stable_contacts,
            'n_stable_contacts': len(stable_contacts)
        }

    def _analyze_contact_stability(self, contact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the stability of contacts over time."""
        stable_contacts = contact_analysis['stable_contacts']
        
        if not stable_contacts:
            return {
                'stability_score': 0.0,
                'contact_persistence': 0.0,
                'stability_distribution': {}
            }
        
        # Calculate stability metrics
        frequencies = [contact['frequency'] for contact in stable_contacts]
        mean_frequency = np.mean(frequencies)
        
        # Contact persistence (how long contacts last on average)
        contact_persistence = mean_frequency
        
        # Stability score based on frequency and distance consistency
        stability_scores = []
        for contact in stable_contacts:
            # Higher frequency and lower distance variability = higher stability
            freq_score = contact['frequency']
            dist_score = 1.0 - min(1.0, contact['std_distance'] / self.contact_cutoff)
            stability_scores.append(freq_score * dist_score)
        
        overall_stability = np.mean(stability_scores)
        
        # Stability distribution
        stability_distribution = {
            'high_stability': len([s for s in stability_scores if s >= 0.7]),
            'medium_stability': len([s for s in stability_scores if 0.3 <= s < 0.7]),
            'low_stability': len([s for s in stability_scores if s < 0.3])
        }
        
        return {
            'stability_score': float(overall_stability),
            'contact_persistence': float(contact_persistence),
            'stability_distribution': stability_distribution,
            'individual_stability_scores': [float(s) for s in stability_scores]
        }

    def _map_binding_interface(self, contact_analysis: Dict[str, Any], 
                              binding_region: Dict[str, Any]) -> Dict[str, Any]:
        """Map the binding interface based on contact patterns."""
        stable_contacts = contact_analysis['stable_contacts']
        
        if not stable_contacts:
            return {
                'interface_atoms': [],
                'interface_residues': [],
                'interface_area': 0.0
            }
        
        # Identify interface atoms (atoms involved in stable contacts)
        interface_atom_indices = set()
        for contact in stable_contacts:
            interface_atom_indices.add(contact['atom1_idx'])
            interface_atom_indices.add(contact['atom2_idx'])
        
        # Get interface atom details
        interface_atoms = []
        interface_residues = set()
        
        for atom in binding_region['atoms']:
            if atom['atom_index'] in interface_atom_indices:
                interface_atoms.append(atom)
                interface_residues.add(atom['resid'])
        
        # Calculate approximate interface area
        # This is a simplified calculation based on number of interface atoms
        interface_area = len(interface_atoms) * 0.1  # Rough estimate in nm²
        
        return {
            'interface_atoms': interface_atoms,
            'interface_residues': sorted(list(interface_residues)),
            'n_interface_atoms': len(interface_atoms),
            'n_interface_residues': len(interface_residues),
            'interface_area': float(interface_area)
        }

    def _assess_binding_strength(self, contact_analysis: Dict[str, Any],
                                contact_stability: Dict[str, Any],
                                binding_interface: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the overall binding strength."""
        # Base binding strength from contact density
        contact_density = contact_analysis['n_stable_contacts'] / max(1, binding_interface['n_interface_atoms'])
        
        # Binding strength from stability
        stability_contribution = contact_stability['stability_score']
        
        # Interface size contribution
        interface_size_score = min(1.0, binding_interface['n_interface_atoms'] / 50.0)  # Normalize to max expected
        
        # Overall binding strength score
        binding_strength_score = (
            contact_density * 0.4 +
            stability_contribution * 0.4 +
            interface_size_score * 0.2
        )
        
        # Categorize binding strength
        if binding_strength_score >= 0.7:
            strength_category = "strong"
        elif binding_strength_score >= 0.4:
            strength_category = "moderate"
        else:
            strength_category = "weak"
        
        return {
            'binding_strength_score': float(binding_strength_score),
            'strength_category': strength_category,
            'contact_density': float(contact_density),
            'stability_contribution': float(stability_contribution),
            'interface_size_contribution': float(interface_size_score)
        }

    def _calculate_contact_quality(self, contact_analysis: Dict[str, Any],
                                 contact_stability: Dict[str, Any],
                                 binding_strength: Dict[str, Any]) -> float:
        """Calculate overall contact quality score (0-1)."""
        # Base score from binding strength
        base_score = binding_strength['binding_strength_score'] * 0.5
        
        # Bonus for contact stability
        stability_bonus = contact_stability['stability_score'] * 0.3
        
        # Bonus for interface size
        interface_bonus = min(0.2, binding_strength['interface_size_contribution'] * 0.2)
        
        total_score = base_score + stability_bonus + interface_bonus
        
        return max(0.0, min(1.0, total_score))

    def _calculate_frame_contacts_mdanalysis(self, universe, atom_indices: List[int], 
                                           frame_idx: int) -> List[Dict[str, Any]]:
        """Calculate contacts for a single frame using MDAnalysis."""
        contacts = []
        
        for i, atom1_idx in enumerate(atom_indices):
            atom1_pos = universe.atoms[atom1_idx].position
            
            for j, atom2_idx in enumerate(atom_indices[i+1:], i+1):
                atom2_pos = universe.atoms[atom2_idx].position
                distance = np.linalg.norm(atom1_pos - atom2_pos)
                
                if distance <= self.contact_cutoff:
                    contacts.append({
                        'atom1_idx': i,
                        'atom2_idx': j,
                        'distance': float(distance)
                    })
        
        return contacts

    def _calculate_frame_contacts_mdtraj(self, trajectory, atom_indices: List[int], 
                                       frame_idx: int) -> List[Dict[str, Any]]:
        """Calculate contacts for a single frame using MDTraj."""
        contacts = []
        
        for i, atom1_idx in enumerate(atom_indices):
            atom1_pos = trajectory.xyz[frame_idx, atom1_idx]
            
            for j, atom2_idx in enumerate(atom_indices[i+1:], i+1):
                atom2_pos = trajectory.xyz[frame_idx, atom2_idx]
                distance = np.linalg.norm(atom1_pos - atom2_pos)
                
                if distance <= self.contact_cutoff:
                    contacts.append({
                        'atom1_idx': i,
                        'atom2_idx': j,
                        'distance': float(distance)
                    })
        
        return contacts 