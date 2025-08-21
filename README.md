# Traj_QC - Trajectory Quality Control System

A comprehensive Python package for GROMACS molecular dynamics trajectory analysis and quality assessment. The system provides a modular, configurable framework covering all major aspects of trajectory quality analysis.

## Features

- **Modular Design**: Support for multiple quality assessment metrics, can be independently enabled/disabled
- **Configuration Driven**: Flexible selection of assessment modules through YAML configuration files
- **GROMACS Compatible**: Direct reading of GROMACS trajectory files (.xtc, .trr, .gro, .pdb)
- **Dual Backend Support**: Works with both MDAnalysis and MDTraj libraries
- **Multi-metric Assessment**: Comprehensive analysis covering all major MD aspects
- **Report Generation**: Automatic generation of quality assessment reports and visualizations

## Analysis Categories

The system is organized into 6 main analysis categories:

### 1. Structural Stability Analysis
- **RMSD**: Root Mean Square Deviation with stability scoring and convergence analysis
- **RMSF**: Root Mean Square Fluctuation with flexible/rigid region identification

### 2. Energy Analysis
- **Total Energy**: Energy conservation analysis with drift detection and correlation analysis
- **Temperature/Pressure**: Temperature and pressure stability with ensemble validation

### 3. Binding State Analysis
- **Contact Area**: Contact area calculation with interface stability assessment
- **Contact Energy**: Interaction energy analysis with binding thermodynamics

### 4. Convergence Analysis
- **RMSD Convergence**: Convergence detection with stable period identification
- **Trajectory Clustering**: Conformational clustering with quality assessment

### 5. Physical Rationality Analysis
- **Atomic Displacement**: Displacement analysis with unphysical movement detection
- **Bond Geometry**: Bond length and angle validation

### 6. Binding Region Analysis
- **Binding Region RMSF**: Local RMSF analysis focused on binding regions
- **Binding Region Contact**: Local contact analysis with interface mapping

## Installation

```bash
# Clone the repository
git clone https://github.com/myuxu95/Traj_QC.git
cd Traj_QC

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Python API Usage

```python
from traj_qc import TrajectoryAssessor

# Initialize with configuration
assessor = TrajectoryAssessor("config/trajectory_quality_config.yaml")

# Load trajectory
assessor.load_trajectory("trajectory.xtc", "topology.pdb")

# Run assessment
results = assessor.run_assessment()

# Access results
rmsd_results = results['structural_stability']['rmsd']
energy_results = results['energy_analysis']['total_energy']
```

### Command Line Usage

```bash
# Basic usage
python traj_qc_cli.py --trajectory md.xtc --topology protein.pdb

# Custom configuration
python traj_qc_cli.py --config my_config.yaml --trajectory md.xtc

# Specific metrics only
python traj_qc_cli.py --trajectory md.xtc --metrics structural_stability,energy_analysis

# Custom output directory
python traj_qc_cli.py --trajectory md.xtc --output results/
```

## Configuration

The system uses YAML configuration files to control which metrics to run and their parameters. See `config/trajectory_quality_config.yaml` for a complete example.

```yaml
metrics:
  structural_stability:
    rmsd:
      enabled: true
      parameters:
        reference_frame: 0
        selection: "protein and name CA"
    
    rmsf:
      enabled: true
      parameters:
        selection: "protein and name CA"
  
  energy_analysis:
    total_energy:
      enabled: true
      parameters:
        energy_file: "energy.xvg"
        energy_columns: ["Potential", "Kinetic", "Total"]
```

## Project Structure

```
Traj_QC/
├── traj_qc/                    # Main package
│   ├── core/                  # Core functionality
│   │   ├── base_metric.py    # Abstract base class for metrics
│   │   └── assessor.py       # Main assessment orchestrator
│   ├── metrics/               # Analysis metrics
│   │   ├── structural_stability.py      # RMSD, RMSF
│   │   ├── energy_analysis.py           # Energy analysis
│   │   ├── binding_analysis.py          # Contact analysis
│   │   ├── convergence_analysis.py      # Convergence, clustering
│   │   ├── physical_rationality.py     # Physical validation
│   │   └── binding_region_analysis.py  # Binding region analysis
│   ├── io/                    # Input/output handling
│   │   └── trajectory_reader.py        # Trajectory file reader
│   ├── utils/                 # Utility functions
│   │   ├── plotting.py        # Visualization functions
│   │   ├── reporting.py       # Report generation
│   │   └── validation.py      # Data validation
│   └── config/                # Configuration management
│       └── manager.py         # Configuration manager
├── config/                     # Configuration files
│   └── trajectory_quality_config.yaml
├── examples/                   # Usage examples
│   └── basic_usage.py
├── tests/                     # Test files
│   └── test_basic_functionality.py
├── setup.py                   # Package installation
├── requirements.txt            # Dependencies
├── traj_qc_cli.py            # CLI interface
└── README.md                  # This file
```

## Dependencies

- **Core Scientific Computing**: numpy, scipy, pandas
- **MD Analysis**: MDAnalysis, mdtraj
- **Visualization**: matplotlib, seaborn, plotly
- **Configuration**: PyYAML
- **Testing**: pytest

## Current Status

### Fully Implemented
- All 6 main analysis categories with complete metric implementations
- Core system architecture and infrastructure
- Configuration management and CLI interface
- Basic plotting and visualization capabilities
- Example scripts and documentation

### Partially Implemented
- Report generation (HTML/PDF) - structure in place, needs full implementation
- Dynamic metric loading - structure in place, needs implementation
- EDR file reading - placeholder methods exist, needs robust implementation

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=traj_qc

# Run specific test file
pytest tests/test_basic_functionality.py -v
```

### Adding New Metrics

1. Create a new metric class inheriting from `BaseMetric`
2. Implement the required `calculate()` and `validate_input()` methods
3. Add the metric to the appropriate category in `traj_qc/metrics/__init__.py`
4. Update the configuration template
5. Add tests for the new metric

### Example Custom Metric

```python
from traj_qc.core.base_metric import BaseMetric

class CustomMetric(BaseMetric):
    def __init__(self, parameter1: str = "default"):
        super().__init__(
            name="custom_metric",
            description="Description of what this metric measures"
        )
        self.parameter1 = parameter1
    
    def validate_input(self, trajectory_data):
        # Validate input data
        return True
    
    def calculate(self, trajectory_data):
        # Implement metric calculation
        result = self._calculate_metric(trajectory_data)
        
        self.results = {
            'metric_value': result,
            'parameter1': self.parameter1
        }
        self.is_calculated = True
        return self.results
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Traj_QC in your research, please cite:

```bibtex
@software{traj_qc,
  title={Traj_QC: Trajectory Quality Control System},
  author={xmy},
  year={2025},
  url={https://github.com/myuxu95/Traj_QC}
}
```

## Support

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: See the examples and configuration files
- **Questions**: Open a GitHub discussion

## Acknowledgments

- Built on top of MDAnalysis and MDTraj libraries
- Inspired by the need for comprehensive trajectory quality assessment
- Developed for the computational chemistry and molecular dynamics community
