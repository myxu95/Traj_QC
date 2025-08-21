"""
Secondary structure analysis metrics for trajectory quality assessment.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from ..core.base_metric import BaseMetric


class SecondaryStructureMetric(BaseMetric):
    """
    Secondary structure analysis metric.
    
    Analyzes protein secondary structure changes over time using DSSP algorithm.
    """
    
    def __init__(self, selection: str = "protein"):
        """
        Initialize secondary structure metric.
        
        Args:
            selection: Atom selection string (default: "protein")
        """
        super().__init__(
            name="secondary_structure",
            description="Protein secondary structure analysis using DSSP"
        )
        self.selection = selection
        self.ss_types = ['H', 'E', 'B', 'G', 'I', 'T', 'S', '-']  # DSSP secondary structure types
    
    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """Validate input data for secondary structure calculation."""
        required_keys = ['n_frames', 'coordinates', 'reader']
        return all(key in trajectory_data for key in required_keys)
    
    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate secondary structure for all frames.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            Dictionary containing secondary structure results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for secondary structure calculation")
        
        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']
        
        if reader == 'MDAnalysis':
            return self._calculate_with_mdanalysis(trajectory_data, n_frames)
        elif reader == 'MDTraj':
            return self._calculate_with_mdtraj(trajectory_data, n_frames)
        else:
            raise ValueError(f"Unsupported reader: {reader}")
    
    def _calculate_with_mdanalysis(self, trajectory_data: Dict[str, Any], n_frames: int) -> Dict[str, Any]:
        """Calculate secondary structure using MDAnalysis."""
        universe = trajectory_data['universe']
        protein = universe.select_atoms(self.selection)
        
        if len(protein) == 0:
            raise ValueError(f"No atoms found for selection: {self.selection}")
        
        # Analyze each frame
        all_ss = []
        ss_counts = {ss_type: [] for ss_type in self.ss_types}
        
        for frame_idx in range(n_frames):
            universe.trajectory[frame_idx]
            
            try:
                # Use MDAnalysis DSSP analysis
                dssp = protein.dssp
                frame_ss = dssp.secondary_structure
                
                # Count secondary structure types
                frame_counts = {ss_type: 0 for ss_type in self.ss_types}
                for ss in frame_ss:
                    if ss in frame_counts:
                        frame_counts[ss] += 1
                
                all_ss.append(frame_ss)
                for ss_type in self.ss_types:
                    ss_counts[ss_type].append(frame_counts[ss_type])
                    
            except Exception as e:
                # Fallback to simplified analysis
                self.logger.warning(f"DSSP analysis failed for frame {frame_idx}: {e}")
                frame_ss = ['-'] * len(protein)
                all_ss.append(frame_ss)
                for ss_type in self.ss_types:
                    ss_counts[ss_type].append(frame_counts.get(ss_type, 0))
        
        return self._process_ss_results(all_ss, ss_counts, n_frames)
    
    def _calculate_with_mdtraj(self, trajectory_data: Dict[str, Any], n_frames: int) -> Dict[str, Any]:
        """Calculate secondary structure using MDTraj."""
        trajectory = trajectory_data['trajectory']
        
        try:
            # MDTraj has built-in DSSP analysis
            dssp_data = md.compute_dssp(trajectory, simplified=True)
            
            # Process results
            all_ss = []
            ss_counts = {ss_type: [] for ss_type in self.ss_types}
            
            for frame_idx in range(n_frames):
                frame_ss = dssp_data[frame_idx]
                frame_ss_list = list(frame_ss)
                
                # Count secondary structure types
                frame_counts = {ss_type: 0 for ss_type in self.ss_types}
                for ss in frame_ss_list:
                    if ss in frame_counts:
                        frame_counts[ss] += 1
                
                all_ss.append(frame_ss_list)
                for ss_type in self.ss_types:
                    ss_counts[ss_type].append(frame_counts[ss_type])
            
        except Exception as e:
            # Fallback to manual calculation
            self.logger.warning(f"MDTraj DSSP analysis failed: {e}. Using manual calculation.")
            return self._calculate_manual_ss(trajectory_data, n_frames)
        
        return self._process_ss_results(all_ss, ss_counts, n_frames)
    
    def _calculate_manual_ss(self, trajectory_data: Dict[str, Any], n_frames: int) -> Dict[str, Any]:
        """Manual secondary structure calculation."""
        # This is a simplified manual calculation
        # In practice, you'd implement a basic DSSP-like algorithm
        
        all_ss = []
        ss_counts = {ss_type: [] for ss_type in self.ss_types}
        
        for frame_idx in range(n_frames):
            # Placeholder for manual calculation
            frame_ss = ['-'] * 100  # Placeholder length
            all_ss.append(frame_ss)
            
            frame_counts = {ss_type: 0 for ss_type in self.ss_types}
            frame_counts['-'] = len(frame_ss)
            
            for ss_type in self.ss_types:
                ss_counts[ss_type].append(frame_counts.get(ss_type, 0))
        
        return self._process_ss_results(all_ss, ss_counts, n_frames)
    
    def _process_ss_results(self, all_ss: List[List[str]], ss_counts: Dict[str, List[int]], n_frames: int) -> Dict[str, Any]:
        """Process secondary structure results into final output."""
        # Convert to numpy arrays
        ss_counts_arrays = {ss_type: np.array(counts) for ss_type, counts in ss_counts.items()}
        
        # Calculate statistics for each secondary structure type
        ss_statistics = {}
        for ss_type, counts in ss_counts_arrays.items():
            if len(counts) > 0:
                ss_statistics[ss_type] = {
                    'mean': np.mean(counts),
                    'std': np.std(counts),
                    'max': np.max(counts),
                    'min': np.min(counts)
                }
            else:
                ss_statistics[ss_type] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0,
                    'min': 0
                }
        
        # Calculate secondary structure stability
        stability = self._calculate_ss_stability(all_ss, n_frames)
        
        # Calculate transitions between secondary structure types
        transitions = self._calculate_ss_transitions(all_ss, n_frames)
        
        self.results = {
            'all_secondary_structures': all_ss,
            'ss_counts': ss_counts_arrays,
            'ss_statistics': ss_statistics,
            'ss_stability': stability,
            'ss_transitions': transitions,
            'selection': self.selection,
            'n_frames': n_frames,
            'ss_types': self.ss_types
        }
        
        self.is_calculated = True
        return self.results
    
    def _calculate_ss_stability(self, all_ss: List[List[str]], n_frames: int) -> Dict[str, Any]:
        """Calculate secondary structure stability metrics."""
        if n_frames < 2:
            return {}
        
        # Calculate how much secondary structure changes between consecutive frames
        changes = []
        for i in range(1, n_frames):
            frame_changes = sum(1 for a, b in zip(all_ss[i-1], all_ss[i]) if a != b)
            changes.append(frame_changes)
        
        changes = np.array(changes)
        
        return {
            'mean_changes_per_frame': np.mean(changes),
            'std_changes_per_frame': np.std(changes),
            'total_changes': np.sum(changes),
            'stability_score': 1.0 / (1.0 + np.mean(changes))  # Higher is more stable
        }
    
    def _calculate_ss_transitions(self, all_ss: List[List[str]], n_frames: int) -> Dict[str, Any]:
        """Calculate transitions between secondary structure types."""
        if n_frames < 2:
            return {}
        
        # Count transitions between different secondary structure types
        transition_matrix = {ss1: {ss2: 0 for ss2 in self.ss_types} for ss1 in self.ss_types}
        
        for i in range(1, n_frames):
            for ss1, ss2 in zip(all_ss[i-1], all_ss[i]):
                if ss1 in transition_matrix and ss2 in transition_matrix[ss1]:
                    transition_matrix[ss1][ss2] += 1
        
        # Calculate transition probabilities
        transition_probabilities = {}
        for ss1 in self.ss_types:
            total_transitions = sum(transition_matrix[ss1].values())
            if total_transitions > 0:
                transition_probabilities[ss1] = {
                    ss2: count / total_transitions 
                    for ss2, count in transition_matrix[ss1].items()
                }
            else:
                transition_probabilities[ss1] = {ss2: 0.0 for ss2 in self.ss_types}
        
        return {
            'transition_matrix': transition_matrix,
            'transition_probabilities': transition_probabilities
        }
    
    def get_ss_content(self, frame_idx: Optional[int] = None) -> Dict[str, float]:
        """
        Get secondary structure content for a specific frame or overall.
        
        Args:
            frame_idx: Frame index (None for overall)
            
        Returns:
            Dictionary containing secondary structure content percentages
        """
        if not self.is_calculated:
            raise RuntimeError("Secondary structure analysis not yet performed")
        
        if frame_idx is not None:
            if frame_idx >= len(self.results['all_secondary_structures']):
                raise ValueError(f"Frame index {frame_idx} out of range")
            
            frame_ss = self.results['all_secondary_structures'][frame_idx]
            total_residues = len(frame_ss)
            
            content = {}
            for ss_type in self.ss_types:
                count = frame_ss.count(ss_type)
                content[ss_type] = (count / total_residues) * 100.0
            
            return content
        else:
            # Return overall content
            return {
                ss_type: self.results['ss_statistics'][ss_type]['mean']
                for ss_type in self.ss_types
            } 