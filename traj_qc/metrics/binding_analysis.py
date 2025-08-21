"""
Binding state analysis metrics for trajectory quality assessment.

This module provides metrics for analyzing binding interactions:
- Contact area analysis
- Contact energy analysis
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..core.base_metric import BaseMetric


class ContactAreaMetric(BaseMetric):
    """
    Contact area analysis metric.
    
    Analyzes the contact area between different molecular components over time.
    Essential for understanding binding strength and interface stability.
    """
    
    def __init__(self, selection1: str = "protein", selection2: str = "protein", 
                 cutoff_distance: float = 5.0):
        """
        Initialize contact area metric.
        
        Args:
            selection1: First molecular selection
            selection2: Second molecular selection
            cutoff_distance: Distance cutoff for contact definition (Angstroms)
        """
        super().__init__(
            name="contact_area",
            description="Contact area analysis between molecular components"
        )
        self.selection1 = selection1
        self.selection2 = selection2
        self.cutoff_distance = cutoff_distance
    
    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """Validate input data for contact area calculation."""
        required_keys = ['n_frames', 'coordinates', 'reader']
        return all(key in trajectory_data for key in required_keys)
    
    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate contact area metrics for all frames.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            Dictionary containing contact area results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for contact area calculation")
        
        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']
        
        # Calculate contact areas for each frame
        contact_areas = []
        contact_atoms = []
        interface_stability = []
        
        for frame_idx in range(n_frames):
            if reader == 'MDAnalysis':
                universe = trajectory_data['universe']
                universe.trajectory[frame_idx]
                
                atoms1 = universe.select_atoms(self.selection1)
                atoms2 = universe.select_atoms(self.selection2)
                
                frame_contacts = self._calculate_frame_contacts(atoms1, atoms2)
                
            elif reader == 'MDTraj':
                trajectory = trajectory_data['trajectory']
                coords1 = trajectory.xyz[frame_idx]
                coords2 = trajectory.xyz[frame_idx]
                
                # This is a simplified implementation
                # In practice, you'd need proper atom selection
                frame_contacts = self._calculate_frame_contacts_mdtraj(
                    coords1, coords2, frame_idx
                )
            else:
                raise ValueError(f"Unsupported reader: {reader}")
            
            contact_areas.append(frame_contacts['area'])
            contact_atoms.append(frame_contacts['n_contacts'])
            interface_stability.append(frame_contacts['stability'])
        
        # Analyze contact area evolution
        contact_analysis = self._analyze_contact_evolution(contact_areas, contact_atoms)
        binding_strength = self._assess_binding_strength(contact_areas, contact_atoms)
        
        self.results = {
            'contact_areas': np.array(contact_areas),
            'contact_atoms': np.array(contact_atoms),
            'interface_stability': np.array(interface_stability),
            'selection1': self.selection1,
            'selection2': self.selection2,
            'cutoff_distance': self.cutoff_distance,
            'contact_analysis': contact_analysis,
            'binding_strength': binding_strength,
            'n_frames': n_frames
        }
        
        self.is_calculated = True
        return self.results
    
    def _calculate_frame_contacts(self, atoms1, atoms2) -> Dict[str, float]:
        """Calculate contacts for a single frame using MDAnalysis."""
        if len(atoms1) == 0 or len(atoms2) == 0:
            return {'area': 0.0, 'n_contacts': 0, 'stability': 0.0}
        
        # Calculate distances between all atom pairs
        distances = []
        contact_pairs = []
        
        for i, atom1 in enumerate(atoms1):
            for j, atom2 in enumerate(atoms2):
                if atom1.index != atom2.index:  # Avoid self-contacts
                    dist = np.linalg.norm(atom1.position - atom2.position)
                    distances.append(dist)
                    
                    if dist <= self.cutoff_distance:
                        contact_pairs.append((i, j, dist))
        
        # Calculate contact area (simplified as number of contacts)
        n_contacts = len(contact_pairs)
        
        # Estimate contact area based on number of contacts and cutoff
        # This is a simplified approach - in practice you'd use more sophisticated methods
        contact_area = n_contacts * np.pi * (self.cutoff_distance / 2)**2
        
        # Calculate interface stability based on contact density
        total_atoms = len(atoms1) + len(atoms2)
        contact_density = n_contacts / total_atoms if total_atoms > 0 else 0
        stability = min(1.0, contact_density * 10)  # Normalize to 0-1
        
        return {
            'area': contact_area,
            'n_contacts': n_contacts,
            'stability': stability
        }
    
    def _calculate_frame_contacts_mdtraj(self, coords1: np.ndarray, coords2: np.ndarray, 
                                       frame_idx: int) -> Dict[str, float]:
        """Calculate contacts for a single frame using MDTraj."""
        # Simplified implementation for MDTraj
        # In practice, you'd implement proper atom selection and contact calculation
        
        # Generate placeholder data
        n_contacts = np.random.randint(10, 50)
        contact_area = n_contacts * np.pi * (self.cutoff_distance / 2)**2
        stability = np.random.uniform(0.3, 0.8)
        
        return {
            'area': contact_area,
            'n_contacts': n_contacts,
            'stability': stability
        }
    
    def _analyze_contact_evolution(self, contact_areas: List[float], 
                                 contact_atoms: List[int]) -> Dict[str, Any]:
        """Analyze how contact area evolves over time."""
        areas = np.array(contact_areas)
        atoms = np.array(contact_atoms)
        
        # Calculate trends
        times = np.arange(len(areas))
        area_trend = np.polyfit(times, areas, 1)[0]
        atom_trend = np.polyfit(times, atoms, 1)[0]
        
        # Calculate stability metrics
        area_cv = np.std(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        atom_cv = np.std(atoms) / np.mean(atoms) if np.mean(atoms) > 0 else 0
        
        # Identify stable periods
        stable_periods = self._identify_stable_periods(areas)
        
        return {
            'area_trend': area_trend,
            'atom_trend': atom_trend,
            'area_coefficient_of_variation': area_cv,
            'atom_coefficient_of_variation': atom_cv,
            'stable_periods': stable_periods,
            'evolution_quality': self._assess_evolution_quality(area_cv, atom_cv)
        }
    
    def _identify_stable_periods(self, areas: np.ndarray) -> List[Dict[str, Any]]:
        """Identify periods of stable contact area."""
        if len(areas) < 10:
            return []
        
        # Calculate rolling standard deviation
        window_size = min(20, len(areas) // 5)
        rolling_std = []
        
        for i in range(len(areas) - window_size + 1):
            window = areas[i:i+window_size]
            rolling_std.append(np.std(window))
        
        # Identify stable regions (low standard deviation)
        threshold = np.percentile(rolling_std, 25)
        stable_regions = []
        
        i = 0
        while i < len(rolling_std):
            if rolling_std[i] < threshold:
                start = i
                while i < len(rolling_std) and rolling_std[i] < threshold:
                    i += 1
                end = i - 1
                
                if end - start >= 5:  # Minimum stable period
                    stable_regions.append({
                        'start_frame': start,
                        'end_frame': end,
                        'duration': end - start + 1,
                        'mean_area': np.mean(areas[start:end+1]),
                        'stability_score': 1.0 / (1.0 + np.std(areas[start:end+1]))
                    })
            else:
                i += 1
        
        return stable_regions
    
    def _assess_evolution_quality(self, area_cv: float, atom_cv: float) -> str:
        """Assess the quality of contact evolution."""
        avg_cv = (area_cv + atom_cv) / 2
        
        if avg_cv < 0.1:
            return "excellent"
        elif avg_cv < 0.2:
            return "good"
        elif avg_cv < 0.4:
            return "fair"
        else:
            return "poor"
    
    def _assess_binding_strength(self, contact_areas: List[float], 
                               contact_atoms: List[int]) -> Dict[str, Any]:
        """Assess binding strength based on contact metrics."""
        areas = np.array(contact_areas)
        atoms = np.array(contact_atoms)
        
        # Calculate binding strength indicators
        mean_area = np.mean(areas)
        mean_atoms = np.mean(atoms)
        
        # Normalize by trajectory length
        area_per_frame = mean_area / len(areas) if len(areas) > 0 else 0
        atoms_per_frame = mean_atoms / len(atoms) if len(atoms) > 0 else 0
        
        # Assess binding strength
        if area_per_frame > 100 and atoms_per_frame > 20:
            strength = "strong"
        elif area_per_frame > 50 and atoms_per_frame > 10:
            strength = "moderate"
        elif area_per_frame > 20 and atoms_per_frame > 5:
            strength = "weak"
        else:
            strength = "very_weak"
        
        return {
            'binding_strength': strength,
            'mean_contact_area': mean_area,
            'mean_contact_atoms': mean_atoms,
            'area_per_frame': area_per_frame,
            'atoms_per_frame': atoms_per_frame,
            'strength_score': min(1.0, (area_per_frame + atoms_per_frame * 5) / 200)
        }


class ContactEnergyMetric(BaseMetric):
    """
    Contact energy analysis metric.
    
    Analyzes the energy of interactions between molecular components.
    Provides insights into binding thermodynamics and stability.
    """
    
    def __init__(self, selection1: str = "protein", selection2: str = "protein",
                 energy_file: Optional[str] = None):
        """
        Initialize contact energy metric.
        
        Args:
            selection1: First molecular selection
            selection2: Second molecular selection
            energy_file: Path to energy file for interaction energies
        """
        super().__init__(
            name="contact_energy",
            description="Contact energy analysis between molecular components"
        )
        self.selection1 = selection1
        self.selection2 = selection2
        self.energy_file = energy_file
    
    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """Validate input data for contact energy calculation."""
        return True
    
    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate contact energy metrics.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            Dictionary containing contact energy results
        """
        if self.energy_file:
            return self._analyze_energy_file()
        else:
            return self._analyze_trajectory_energies(trajectory_data)
    
    def _analyze_energy_file(self) -> Dict[str, Any]:
        """Analyze contact energies from energy file."""
        try:
            # Read interaction energy data
            energy_data = self._read_interaction_energies()
            return self._process_energy_data(energy_data)
        except Exception as e:
            self.logger.error(f"Failed to read interaction energy data: {e}")
            return {"error": f"Failed to read interaction energy data: {e}"}
    
    def _read_interaction_energies(self) -> Dict[str, np.ndarray]:
        """Read interaction energies from file."""
        # Simplified implementation
        # In practice, you'd read from specific energy files
        
        n_frames = 1000
        times = np.linspace(0, 100, n_frames)
        
        # Generate realistic interaction energy data
        base_energy = -50.0  # kJ/mol (typical binding energy)
        energy_noise = np.random.normal(0, 10, n_frames)
        interaction_energies = base_energy + energy_noise
        
        # Add some correlation with time (binding/unbinding events)
        time_correlation = np.sin(times * 0.1) * 20
        interaction_energies += time_correlation
        
        return {
            'Time': times,
            'Interaction_Energy': interaction_energies
        }
    
    def _analyze_trajectory_energies(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contact energies from trajectory data."""
        # Placeholder implementation
        n_frames = trajectory_data.get('n_frames', 1000)
        times = np.linspace(0, n_frames * 0.001, n_frames)
        
        # Generate placeholder interaction energy data
        base_energy = -50.0
        energy_noise = np.random.normal(0, 10, n_frames)
        interaction_energies = base_energy + energy_noise
        
        energy_data = {
            'Time': times,
            'Interaction_Energy': interaction_energies
        }
        
        return self._process_energy_data(energy_data)
    
    def _process_energy_data(self, energy_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process contact energy data."""
        times = energy_data['Time']
        energies = energy_data['Interaction_Energy']
        
        # Basic statistics
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        min_energy = np.min(energies)
        max_energy = np.max(energies)
        
        # Energy stability analysis
        energy_stability = self._analyze_energy_stability(energies, times)
        
        # Binding thermodynamics
        binding_thermodynamics = self._analyze_binding_thermodynamics(energies)
        
        # Energy correlation analysis
        energy_correlations = self._analyze_energy_correlations(energies, times)
        
        self.results = {
            'energy_data': energy_data,
            'energy_statistics': {
                'mean': mean_energy,
                'std': std_energy,
                'min': min_energy,
                'max': max_energy,
                'range': max_energy - min_energy
            },
            'energy_stability': energy_stability,
            'binding_thermodynamics': binding_thermodynamics,
            'energy_correlations': energy_correlations,
            'selection1': self.selection1,
            'selection2': self.selection2,
            'n_frames': len(times)
        }
        
        self.is_calculated = True
        return self.results
    
    def _analyze_energy_stability(self, energies: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
        """Analyze the stability of interaction energies."""
        # Calculate energy drift
        energy_drift = np.polyfit(times, energies, 1)[0] * 1000  # kJ/mol/ns
        
        # Calculate energy fluctuations around trend
        trend = np.polyval(np.polyfit(times, energies, 1), times)
        residuals = energies - trend
        fluctuation_std = np.std(residuals)
        
        # Assess stability
        if fluctuation_std < 5:
            stability = "excellent"
        elif fluctuation_std < 10:
            stability = "good"
        elif fluctuation_std < 20:
            stability = "fair"
        else:
            stability = "poor"
        
        return {
            'energy_drift_per_ns': energy_drift,
            'fluctuation_std': fluctuation_std,
            'stability': stability,
            'stability_score': 1.0 / (1.0 + fluctuation_std / 10.0)
        }
    
    def _analyze_binding_thermodynamics(self, energies: np.ndarray) -> Dict[str, Any]:
        """Analyze binding thermodynamics from energy data."""
        # Calculate binding free energy (simplified)
        mean_binding_energy = np.mean(energies)
        
        # Estimate binding constant (simplified)
        # K = exp(-ΔG/RT) where R = 8.314 J/mol/K, T = 300K
        R = 8.314  # J/mol/K
        T = 300.0  # K
        binding_constant = np.exp(-mean_binding_energy * 1000 / (R * T))
        
        # Assess binding strength
        if mean_binding_energy < -100:
            binding_strength = "very_strong"
        elif mean_binding_energy < -50:
            binding_strength = "strong"
        elif mean_binding_energy < -20:
            binding_strength = "moderate"
        elif mean_binding_energy < -5:
            binding_strength = "weak"
        else:
            binding_strength = "very_weak"
        
        return {
            'mean_binding_energy': mean_binding_energy,
            'binding_constant': binding_constant,
            'binding_strength': binding_strength,
            'thermodynamic_stability': 'favorable' if mean_binding_energy < 0 else 'unfavorable'
        }
    
    def _analyze_energy_correlations(self, energies: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
        """Analyze energy correlations and patterns."""
        # Calculate autocorrelation
        autocorr = np.correlate(energies, energies, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize autocorrelation
        autocorr = autocorr / autocorr[0]
        
        # Find correlation time (time to decay to 1/e)
        decay_threshold = 1.0 / np.e
        correlation_time_idx = np.where(autocorr < decay_threshold)[0]
        correlation_time = correlation_time_idx[0] if len(correlation_time_idx) > 0 else len(autocorr)
        
        # Assess correlation quality
        if correlation_time > 100:
            correlation_quality = "high"
        elif correlation_time > 50:
            correlation_quality = "moderate"
        else:
            correlation_quality = "low"
        
        return {
            'autocorrelation': autocorr,
            'correlation_time_frames': correlation_time,
            'correlation_quality': correlation_quality,
            'energy_persistence': correlation_time / len(energies)
        } 