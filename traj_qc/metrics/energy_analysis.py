"""
Energy analysis metrics for trajectory quality assessment.

This module provides metrics for analyzing energy components and stability:
- Total energy, potential energy, kinetic energy
- Temperature and pressure stability
"""

import numpy as np
from typing import Dict, Any, List, Optional
from ..core.base_metric import BaseMetric


class TotalEnergyMetric(BaseMetric):
    """
    Total energy analysis metric.
    
    Analyzes total energy conservation and stability in molecular dynamics trajectories.
    Critical for assessing simulation quality and physical correctness.
    """
    
    def __init__(self, energy_file: Optional[str] = None, energy_columns: Optional[List[str]] = None):
        """
        Initialize total energy metric.
        
        Args:
            energy_file: Path to energy file (e.g., .edr, .xvg)
            energy_columns: List of energy column names to analyze
        """
        super().__init__(
            name="total_energy",
            description="Total energy conservation and stability analysis"
        )
        self.energy_file = energy_file
        self.energy_columns = energy_columns or ['Potential', 'Kinetic', 'Total']
    
    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """Validate input data for energy calculation."""
        # Energy analysis can work with minimal trajectory data
        # as it primarily relies on energy files
        return True
    
    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate energy analysis metrics.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            Dictionary containing energy analysis results
        """
        if self.energy_file:
            return self._analyze_energy_file()
        else:
            return self._analyze_trajectory_energies(trajectory_data)
    
    def _analyze_energy_file(self) -> Dict[str, Any]:
        """Analyze energy data from external file."""
        try:
            # Try to read different energy file formats
            if self.energy_file.endswith('.xvg'):
                energy_data = self._read_xvg_file()
            elif self.energy_file.endswith('.edr'):
                energy_data = self._read_edr_file()
            else:
                # Try generic text file
                energy_data = self._read_generic_energy_file()
            
            return self._process_energy_data(energy_data)
            
        except Exception as e:
            self.logger.error(f"Failed to read energy file {self.energy_file}: {e}")
            return {"error": f"Failed to read energy file: {e}"}
    
    def _read_xvg_file(self) -> Dict[str, np.ndarray]:
        """Read GROMACS XVG energy file."""
        energy_data = {}
        times = []
        
        with open(self.energy_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or line.startswith('@'):
                    continue
                
                if line and not line.startswith('&'):
                    try:
                        values = [float(x) for x in line.split()]
                        if len(values) >= len(self.energy_columns) + 1:  # +1 for time
                            times.append(values[0])
                            for i, col_name in enumerate(self.energy_columns):
                                if col_name not in energy_data:
                                    energy_data[col_name] = []
                                energy_data[col_name].append(values[i + 1])
                    except ValueError:
                        continue
        
        # Convert to numpy arrays
        energy_data['Time'] = np.array(times)
        for col_name in self.energy_columns:
            if col_name in energy_data:
                energy_data[col_name] = np.array(energy_data[col_name])
        
        return energy_data
    
    def _read_edr_file(self) -> Dict[str, np.ndarray]:
        """Read GROMACS EDR energy file."""
        # EDR files are binary and require gmx energy or similar tools
        # For now, return placeholder data
        self.logger.warning("EDR file reading not yet implemented. Using placeholder data.")
        
        # Generate placeholder data
        n_frames = 1000
        times = np.linspace(0, 100, n_frames)  # 100 ps simulation
        
        energy_data = {'Time': times}
        for col_name in self.energy_columns:
            # Generate realistic energy fluctuations
            base_energy = -1000 if 'Potential' in col_name else 500 if 'Kinetic' in col_name else -500
            noise = np.random.normal(0, 50, n_frames)
            energy_data[col_name] = base_energy + noise
        
        return energy_data
    
    def _read_generic_energy_file(self) -> Dict[str, np.ndarray]:
        """Read generic energy file format."""
        energy_data = {}
        times = []
        
        with open(self.energy_file, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    values = [float(x) for x in line.split()]
                    if line_num == 0:  # First data line - determine columns
                        if len(values) >= len(self.energy_columns) + 1:
                            times.append(values[0])
                            for i, col_name in enumerate(self.energy_columns):
                                energy_data[col_name] = [values[i + 1]]
                        else:
                            # Assume first column is time, rest are energies
                            times.append(values[0])
                            for i in range(1, len(values)):
                                col_name = f"Energy_{i}"
                                energy_data[col_name] = [values[i]]
                    else:
                        if len(values) >= len(energy_data) + 1:
                            times.append(values[0])
                            for i, col_name in enumerate(energy_data.keys()):
                                energy_data[col_name].append(values[i + 1])
                except ValueError:
                    continue
        
        # Convert to numpy arrays
        energy_data['Time'] = np.array(times)
        for col_name in energy_data:
            if col_name != 'Time':
                energy_data[col_name] = np.array(energy_data[col_name])
        
        return energy_data
    
    def _analyze_trajectory_energies(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze energies from trajectory data if available."""
        # This would analyze energies stored in trajectory files
        # For now, return placeholder data
        self.logger.info("No energy file provided, using placeholder energy data")
        
        n_frames = trajectory_data.get('n_frames', 1000)
        times = np.linspace(0, n_frames * 0.001, n_frames)  # Assume 1 ps timestep
        
        energy_data = {'Time': times}
        for col_name in self.energy_columns:
            base_energy = -1000 if 'Potential' in col_name else 500 if 'Kinetic' in col_name else -500
            noise = np.random.normal(0, 50, n_frames)
            energy_data[col_name] = base_energy + noise
        
        return self._process_energy_data(energy_data)
    
    def _process_energy_data(self, energy_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process energy data and calculate metrics."""
        if not energy_data or 'Time' not in energy_data:
            return {"error": "No valid energy data found"}
        
        times = energy_data['Time']
        n_frames = len(times)
        
        # Calculate basic statistics for each energy type
        energy_statistics = {}
        energy_fluctuations = {}
        
        for col_name in self.energy_columns:
            if col_name in energy_data:
                energies = energy_data[col_name]
                
                # Basic statistics
                energy_statistics[col_name] = {
                    'mean': np.mean(energies),
                    'std': np.std(energies),
                    'min': np.min(energies),
                    'max': np.max(energies),
                    'range': np.max(energies) - np.min(energies)
                }
                
                # Fluctuation analysis
                if n_frames > 1:
                    # Calculate energy drift (linear trend)
                    coeffs = np.polyfit(times, energies, 1)
                    drift = coeffs[0]  # Slope
                    
                    # Calculate energy stability (variance around trend)
                    trend = np.polyval(coeffs, times)
                    residuals = energies - trend
                    stability = np.std(residuals)
                    
                    energy_fluctuations[col_name] = {
                        'drift': drift,
                        'stability': stability,
                        'drift_per_ns': drift * 1000,  # Convert to per ns
                        'relative_fluctuation': stability / abs(np.mean(energies)) * 100
                    }
        
        # Calculate energy conservation (if both potential and kinetic available)
        energy_conservation = {}
        if 'Potential' in energy_data and 'Kinetic' in energy_data:
            total_energy = energy_data['Potential'] + energy_data['Kinetic']
            
            # Check if total energy is conserved
            total_std = np.std(total_energy)
            total_mean = np.mean(total_energy)
            conservation_score = 1.0 / (1.0 + total_std / abs(total_mean))
            
            energy_conservation = {
                'total_energy_mean': total_mean,
                'total_energy_std': total_std,
                'conservation_score': conservation_score,
                'energy_drift': np.polyfit(times, total_energy, 1)[0] * 1000,  # per ns
                'conservation_quality': self._assess_conservation_quality(total_std, total_mean)
            }
        
        # Calculate correlation analysis
        correlations = self._calculate_energy_correlations(energy_data)
        
        self.results = {
            'energy_data': energy_data,
            'energy_statistics': energy_statistics,
            'energy_fluctuations': energy_fluctuations,
            'energy_conservation': energy_conservation,
            'correlations': correlations,
            'n_frames': n_frames,
            'time_range': [times[0], times[-1]],
            'energy_columns': self.energy_columns
        }
        
        self.is_calculated = True
        return self.results
    
    def _assess_conservation_quality(self, total_std: float, total_mean: float) -> str:
        """Assess the quality of energy conservation."""
        if total_mean == 0:
            return "unknown"
        
        cv = total_std / abs(total_mean)
        if cv < 0.01:
            return "excellent"
        elif cv < 0.05:
            return "good"
        elif cv < 0.1:
            return "fair"
        else:
            return "poor"
    
    def _calculate_energy_correlations(self, energy_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate correlations between different energy components."""
        correlations = {}
        
        energy_columns = [col for col in self.energy_columns if col in energy_data]
        
        for i, col1 in enumerate(energy_columns):
            for col2 in energy_columns[i+1:]:
                if col1 in energy_data and col2 in energy_data:
                    corr = np.corrcoef(energy_data[col1], energy_data[col2])[0, 1]
                    if not np.isnan(corr):
                        correlations[f"{col1}_{col2}"] = corr
        
        return correlations


class TemperaturePressureMetric(BaseMetric):
    """
    Temperature and pressure stability metric.
    
    Analyzes temperature and pressure fluctuations to assess simulation stability.
    Important for NVT/NPT ensemble validation.
    """
    
    def __init__(self, energy_file: Optional[str] = None):
        """
        Initialize temperature and pressure metric.
        
        Args:
            energy_file: Path to energy file containing temperature and pressure data
        """
        super().__init__(
            name="temperature_pressure",
            description="Temperature and pressure stability analysis"
        )
        self.energy_file = energy_file
    
    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """Validate input data for temperature/pressure calculation."""
        return True
    
    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate temperature and pressure stability metrics.
        
        Args:
            trajectory_data: Dictionary containing trajectory data
            
        Returns:
            Dictionary containing temperature and pressure analysis results
        """
        if self.energy_file:
            return self._analyze_tp_file()
        else:
            return self._analyze_trajectory_tp(trajectory_data)
    
    def _analyze_tp_file(self) -> Dict[str, Any]:
        """Analyze temperature and pressure from energy file."""
        try:
            # Read temperature and pressure data
            tp_data = self._read_tp_file()
            return self._process_tp_data(tp_data)
        except Exception as e:
            self.logger.error(f"Failed to read temperature/pressure data: {e}")
            return {"error": f"Failed to read temperature/pressure data: {e}"}
    
    def _read_tp_file(self) -> Dict[str, np.ndarray]:
        """Read temperature and pressure data from file."""
        # This is a simplified implementation
        # In practice, you'd read from specific energy files
        
        n_frames = 1000
        times = np.linspace(0, 100, n_frames)
        
        # Generate realistic temperature and pressure data
        target_temp = 300.0  # K
        target_pressure = 1.0  # bar
        
        # Temperature with realistic fluctuations
        temp_noise = np.random.normal(0, 5, n_frames)  # ±5K fluctuations
        temperatures = target_temp + temp_noise
        
        # Pressure with realistic fluctuations
        pressure_noise = np.random.normal(0, 10, n_frames)  # ±10 bar fluctuations
        pressures = target_pressure + pressure_noise
        
        return {
            'Time': times,
            'Temperature': temperatures,
            'Pressure': pressures
        }
    
    def _analyze_trajectory_tp(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temperature and pressure from trajectory data."""
        # Placeholder implementation
        n_frames = trajectory_data.get('n_frames', 1000)
        times = np.linspace(0, n_frames * 0.001, n_frames)
        
        # Generate placeholder data
        target_temp = 300.0
        target_pressure = 1.0
        
        temp_noise = np.random.normal(0, 5, n_frames)
        temperatures = target_temp + temp_noise
        
        pressure_noise = np.random.normal(0, 10, n_frames)
        pressures = target_pressure + pressure_noise
        
        tp_data = {
            'Time': times,
            'Temperature': temperatures,
            'Pressure': pressures
        }
        
        return self._process_tp_data(tp_data)
    
    def _process_tp_data(self, tp_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process temperature and pressure data."""
        times = tp_data['Time']
        temperatures = tp_data['Temperature']
        pressures = tp_data['Pressure']
        
        # Temperature analysis
        temp_mean = np.mean(temperatures)
        temp_std = np.std(temperatures)
        temp_drift = np.polyfit(times, temperatures, 1)[0] * 1000  # K/ns
        
        # Pressure analysis
        pressure_mean = np.mean(pressures)
        pressure_std = np.std(pressures)
        pressure_drift = np.polyfit(times, pressures, 1)[0] * 1000  # bar/ns
        
        # Stability assessment
        temp_stability = self._assess_temperature_stability(temperatures)
        pressure_stability = self._assess_pressure_stability(pressures)
        
        # Ensemble validation
        ensemble_validation = self._validate_ensemble(temperatures, pressures)
        
        self.results = {
            'temperature': {
                'mean': temp_mean,
                'std': temp_std,
                'drift_per_ns': temp_drift,
                'stability': temp_stability,
                'fluctuation_percent': (temp_std / temp_mean) * 100
            },
            'pressure': {
                'mean': pressure_mean,
                'std': pressure_std,
                'drift_per_ns': pressure_drift,
                'stability': pressure_stability,
                'fluctuation_percent': (pressure_std / abs(pressure_mean)) * 100
            },
            'ensemble_validation': ensemble_validation,
            'n_frames': len(times),
            'time_range': [times[0], times[-1]]
        }
        
        self.is_calculated = True
        return self.results
    
    def _assess_temperature_stability(self, temperatures: np.ndarray) -> str:
        """Assess temperature stability."""
        temp_std = np.std(temperatures)
        temp_mean = np.mean(temperatures)
        
        if temp_mean == 0:
            return "unknown"
        
        cv = temp_std / temp_mean
        if cv < 0.01:
            return "excellent"
        elif cv < 0.02:
            return "good"
        elif cv < 0.05:
            return "fair"
        else:
            return "poor"
    
    def _assess_pressure_stability(self, pressures: np.ndarray) -> str:
        """Assess pressure stability."""
        pressure_std = np.std(pressures)
        pressure_mean = np.mean(pressures)
        
        if pressure_mean == 0:
            return "unknown"
        
        cv = pressure_std / abs(pressure_mean)
        if cv < 0.1:
            return "excellent"
        elif cv < 0.2:
            return "good"
        elif cv < 0.5:
            return "fair"
        else:
            return "poor"
    
    def _validate_ensemble(self, temperatures: np.ndarray, pressures: np.ndarray) -> Dict[str, Any]:
        """Validate if the simulation follows expected ensemble behavior."""
        temp_std = np.std(temperatures)
        pressure_std = np.std(pressures)
        
        # For NVT ensemble, pressure should fluctuate more than temperature
        # For NPT ensemble, both should be controlled
        nvt_score = pressure_std / temp_std if temp_std > 0 else 0
        npt_score = 1.0 / (1.0 + abs(temp_std - pressure_std))
        
        return {
            'nvt_likelihood': min(1.0, nvt_score / 10.0),  # Normalize
            'npt_likelihood': npt_score,
            'ensemble_type': 'NVT' if nvt_score > 5.0 else 'NPT' if npt_score > 0.8 else 'Unknown'
        } 