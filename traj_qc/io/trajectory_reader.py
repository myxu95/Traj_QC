"""
Trajectory reader for GROMACS trajectory files.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

try:
    import MDAnalysis as mda
    MDAVAILABLE = True
except ImportError:
    MDAVAILABLE = False
    logging.warning("MDAnalysis not available. Some functionality may be limited.")

try:
    import mdtraj as md
    MDTRAJAVAILABLE = True
except ImportError:
    MDTRAJAVAILABLE = False
    logging.warning("MDTraj not available. Some functionality may be limited.")


class TrajectoryReader:
    """
    Reader for GROMACS trajectory files.
    
    Supports multiple trajectory formats and provides a unified interface
    for accessing trajectory data.
    """
    
    def __init__(self):
        """Initialize the trajectory reader."""
        self.logger = logging.getLogger(__name__)
        self.trajectory = None
        self.topology = None
        
        if not MDAVAILABLE and not MDTRAJAVAILABLE:
            self.logger.error("Neither MDAnalysis nor MDTraj is available. Cannot read trajectories.")
    
    def read_trajectory(self, trajectory_path: str, topology_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Read trajectory data from file.
        
        Args:
            trajectory_path: Path to trajectory file
            topology_path: Path to topology file (optional)
            
        Returns:
            Dictionary containing trajectory data
        """
        trajectory_path = Path(trajectory_path)
        
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
        
        self.logger.info(f"Reading trajectory from: {trajectory_path}")
        
        # Try to determine file format
        file_extension = trajectory_path.suffix.lower()
        
        if file_extension in ['.xtc', '.trr', '.gro', '.pdb']:
            return self._read_with_mdanalysis(trajectory_path, topology_path)
        elif file_extension in ['.h5', '.nc', '.dcd']:
            return self._read_with_mdtraj(trajectory_path, topology_path)
        else:
            # Try both readers
            try:
                return self._read_with_mdanalysis(trajectory_path, topology_path)
            except Exception as e1:
                try:
                    return self._read_with_mdtraj(trajectory_path, topology_path)
                except Exception as e2:
                    raise RuntimeError(f"Failed to read trajectory with both readers. MDAnalysis error: {e1}, MDTraj error: {e2}")
    
    def _read_with_mdanalysis(self, trajectory_path: Path, topology_path: Optional[str] = None) -> Dict[str, Any]:
        """Read trajectory using MDAnalysis."""
        if not MDAVAILABLE:
            raise ImportError("MDAnalysis is not available")
        
        try:
            if topology_path:
                self.trajectory = mda.Universe(topology_path, str(trajectory_path))
            else:
                self.trajectory = mda.Universe(str(trajectory_path))
            
            # Extract basic information
            n_frames = len(self.trajectory.trajectory)
            n_atoms = len(self.trajectory.atoms)
            
            # Get time information
            times = []
            for ts in self.trajectory.trajectory:
                times.append(ts.time)
            
            # Get atom information
            atom_info = {
                'names': self.trajectory.atoms.names,
                'types': self.trajectory.atoms.types,
                'resnames': self.trajectory.atoms.resnames,
                'resids': self.trajectory.atoms.resids,
                'segids': self.trajectory.atoms.segids
            }
            
            # Get coordinates for first frame as reference
            self.trajectory.trajectory[0]
            coordinates = self.trajectory.atoms.positions
            
            trajectory_data = {
                'n_frames': n_frames,
                'n_atoms': n_atoms,
                'times': np.array(times),
                'atom_info': atom_info,
                'coordinates': coordinates,
                'reader': 'MDAnalysis',
                'trajectory_path': str(trajectory_path),
                'topology_path': topology_path,
                'universe': self.trajectory
            }
            
            self.logger.info(f"Successfully read trajectory with MDAnalysis: {n_frames} frames, {n_atoms} atoms")
            return trajectory_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to read trajectory with MDAnalysis: {e}")
    
    def _read_with_mdtraj(self, trajectory_path: Path, topology_path: Optional[str] = None) -> Dict[str, Any]:
        """Read trajectory using MDTraj."""
        if not MDTRAJAVAILABLE:
            raise ImportError("MDTraj is not available")
        
        try:
            if topology_path:
                self.trajectory = md.load(str(trajectory_path), top=topology_path)
            else:
                self.trajectory = md.load(str(trajectory_path))
            
            # Extract basic information
            n_frames = self.trajectory.n_frames
            n_atoms = self.trajectory.n_atoms
            
            # Get time information (MDTraj doesn't always have time)
            try:
                times = self.trajectory.time
            except:
                times = np.arange(n_frames) * 0.001  # Default 1 ps timestep
            
            # Get atom information
            atom_info = {
                'names': [atom.name for atom in self.trajectory.topology.atoms],
                'types': [atom.element.symbol for atom in self.trajectory.topology.atoms],
                'resnames': [res.name for res in self.trajectory.topology.residues],
                'resids': [res.index for res in self.trajectory.topology.residues],
                'segids': ['A' for _ in range(n_atoms)]  # MDTraj doesn't have segids
            }
            
            # Get coordinates for first frame as reference
            coordinates = self.trajectory.xyz[0]
            
            trajectory_data = {
                'n_frames': n_frames,
                'n_atoms': n_atoms,
                'times': times,
                'atom_info': atom_info,
                'coordinates': coordinates,
                'reader': 'MDTraj',
                'trajectory_path': str(trajectory_path),
                'topology_path': topology_path,
                'trajectory': self.trajectory
            }
            
            self.logger.info(f"Successfully read trajectory with MDTraj: {n_frames} frames, {n_atoms} atoms")
            return trajectory_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to read trajectory with MDTraj: {e}")
    
    def get_frame_coordinates(self, frame_index: int) -> np.ndarray:
        """
        Get coordinates for a specific frame.
        
        Args:
            frame_index: Index of the frame to retrieve
            
        Returns:
            Array of coordinates for the specified frame
        """
        if self.trajectory is None:
            raise RuntimeError("No trajectory loaded")
        
        if hasattr(self.trajectory, 'trajectory'):  # MDAnalysis
            self.trajectory.trajectory[frame_index]
            return self.trajectory.atoms.positions
        else:  # MDTraj
            return self.trajectory.xyz[frame_index]
    
    def get_atom_selection(self, selection_string: str) -> np.ndarray:
        """
        Get atom indices based on selection string.
        
        Args:
            selection_string: MDAnalysis-style selection string
            
        Returns:
            Array of atom indices matching the selection
        """
        if self.trajectory is None:
            raise RuntimeError("No trajectory loaded")
        
        if hasattr(self.trajectory, 'trajectory'):  # MDAnalysis
            selection = self.trajectory.select_atoms(selection_string)
            return selection.indices
        else:  # MDTraj
            # Convert MDAnalysis selection to MDTraj selection
            # This is a simplified conversion
            selection = self.trajectory.topology.select(selection_string)
            return selection if selection is not None else np.array([])
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded trajectory.
        
        Returns:
            Dictionary containing trajectory summary information
        """
        if self.trajectory is None:
            return {"status": "no_trajectory_loaded"}
        
        if hasattr(self.trajectory, 'trajectory'):  # MDAnalysis
            return {
                "reader": "MDAnalysis",
                "n_frames": len(self.trajectory.trajectory),
                "n_atoms": len(self.trajectory.atoms),
                "n_residues": len(self.trajectory.residues),
                "n_segments": len(self.trajectory.segments),
                "dimensions": self.trajectory.dimensions if hasattr(self.trajectory, 'dimensions') else None
            }
        else:  # MDTraj
            return {
                "reader": "MDTraj",
                "n_frames": self.trajectory.n_frames,
                "n_atoms": self.trajectory.n_atoms,
                "n_residues": self.trajectory.n_residues,
                "unit_cell_angles": self.trajectory.unitcell_angles[0] if self.trajectory.unitcell_angles is not None else None,
                "unit_cell_lengths": self.trajectory.unitcell_lengths[0] if self.trajectory.unitcell_lengths is not None else None
            } 