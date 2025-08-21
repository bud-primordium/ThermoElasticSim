# ThermoElasticSim

**Languages**: [中文](README.md) | English

## Project Overview

**ThermoElasticSim** is a molecular dynamics simulation tool specifically designed for calculating elastic constants of aluminum (Al) at zero and finite temperatures. Based on the EAM (Embedded Atom Method) potential, this project combines Python's ease of use with C++'s high-performance computing capabilities, employing explicit deformation and stress fluctuation methods to accurately calculate elastic properties of materials through structural optimization and molecular dynamics (MD) simulations.

The project implements multiple MD ensembles using an operator splitting architecture, including NVE, NVT (Berendsen, Andersen, Nosé-Hoover chain), and NPT (MTK), providing a reliable statistical mechanics foundation for finite-temperature elastic constant calculations.

## Key Features

### Implemented Features

1. **Zero-Temperature Elastic Constants (FCC Aluminum)**
   - Determination of equilibrium lattice configuration
   - Independent strain application and stress response calculation
   - Elastic constants calculation (C11, C12, C44)
   - Multiple system sizes and convergence analysis

2. **Finite-Temperature Elastic Constants (FCC Aluminum)**
   - MTK-NPT pre-equilibration for finite-temperature equilibrium states
   - Nosé-Hoover chain NVT sampling of stress fluctuations
   - Statistical averaging for temperature-dependent elastic constants

3. **Molecular Dynamics Ensembles**
   - NVE: Operator splitting Velocity-Verlet integration
   - NVT: Berendsen weak coupling, Andersen stochastic collision, Nosé-Hoover chain thermostats
   - NPT: Martyna-Tobias-Klein reversible integrator

4. **Computational Architecture**
   - C++/Python hybrid programming with core calculations in C++
   - EAM potential and virial stress tensor calculations
   - Periodic boundary conditions with minimum image convention
   - Multiple numerical optimizers

### Features in Development

- Configuration file system (YAML format)
- Elastic wave propagation simulation
- Diamond material support

## Potential Model

The project employs a validated EAM (Embedded Atom Method) potential for aluminum molecular dynamics simulations:

**EAM_Dynamo_MendelevKramerBecker_2008_Al__MO_106969701023_006**

- **OpenKIM Database**: [MO_106969701023_006](https://openkim.org/id/EAM_Dynamo_MendelevKramerBecker_2008_Al__MO_106969701023_006)
- **Theoretical Basis**: Mendelev MI, Kramer MJ, Becker CA, Asta M. *Analysis of semi-empirical interatomic potentials appropriate for simulation of crystalline and liquid Al and Cu*. Philosophical Magazine. 2008;88(12):1723–50. doi:10.1080/14786430802206482
- **Applicable Range**: Structural and dynamic properties of crystalline and liquid aluminum
- **Validation Accuracy**: Zero-temperature elastic constants error < 1% compared to literature values, good agreement with experimental values at finite temperatures

## Directory Structure

```plaintext
ThermoElasticSim/
├── pyproject.toml          # Project configuration (dependency management, build settings)
├── CMakeLists.txt          # C++ build configuration
├── README.md
├── src/
│   └── thermoelasticsim/
│       ├── _cpp/           # C++ source code and pybind11 bindings
│       │   ├── bindings/   # Modular binding files
│       │   └── *.cpp       # Core computation implementation
│       ├── core/           # Core data structures
│       ├── potentials/     # Potential models
│       ├── elastic/        # Elastic constants calculation
│       ├── md/             # Molecular dynamics
│       └── utils/          # Utility functions
├── tests/                  # Test files (mirroring src structure)
├── examples/               # Usage examples
└── docs/                   # Documentation
```

## Installation Guide

### Prerequisites

- **Python 3.9+**
- **C++ compiler** (supporting C++11)
- **uv** (recommended) or pip

### Quick Installation

```bash
# 1. Clone repository
git clone https://github.com/bud-primordium/ThermoElasticSim.git
cd ThermoElasticSim

# 2. Clean old build artifacts (if any)
rm -rf .venv build .cmake src/thermoelasticsim/_cpp_core*.so

# 3. Create virtual environment
uv venv

# 4. Install project (editable mode)
uv pip install -e .

# 5. Install test dependencies (if running tests)
uv pip install pytest

# 6. Verify installation
uv run python -c "import thermoelasticsim._cpp_core; print('✓ Installation successful')"
```

### Running Tests

```bash
uv run pytest
```

### Installation Notes

- **Automated Build**: Uses `scikit-build-core` + `pybind11` for automatic C++ extension compilation
- **Known Issue**: Due to scikit-build-core compatibility issues with uv, currently only editable installation (`-e`) is supported
- **Important**: All commands must be prefixed with `uv run` to ensure correct Python environment usage

## Usage Instructions

### Running Examples

**Zero-Temperature Elastic Constants Calculation**:
```bash
uv run python examples/zero_temp_al_benchmark.py
```

**Finite-Temperature Elastic Constants Calculation**:
```bash
uv run python examples/finite_temp_al_benchmark.py
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific tests
uv run pytest tests/potentials/

# Show detailed output
uv run pytest -v
```

## Configuration Files

The project uses YAML format configuration files `tests/config.yaml` for parameter management. Here's an example configuration (**currently under testing**):

```yaml
# config.yaml

unit_cell:
  lattice_vectors: 
    - [3.615, 0.0, 0.0]
    - [0.0, 3.615, 0.0]
    - [0.0, 0.0, 3.615]
  atoms:
    - {symbol: 'Al', mass: 26.9815, position: [0.0, 0.0, 0.0]}
    - {symbol: 'Al', mass: 26.9815, position: [1.8075, 1.8075, 1.8075]}

potential:
  type: 'LennardJones'
  parameters:
    epsilon: 0.0103
    sigma: 3.405
  cutoff: 5.0

optimizer:
  method: 'ConjugateGradient'

deformation:
  delta: 0.01

stress_evaluator:
  type: 'LennardJones'

md_simulation:
  temperature: 300
  pressure: 0.0
  timestep: 1.0e-3
  thermostat: 'Nosé-Hoover'
  barostat: 'NoBarostat'
  steps: 10000
```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a branch**

   ```bash
   git checkout -b feature/new-feature-name
   ```

3. **Commit changes**

   ```bash
   git commit -m "Add new feature"
   ```

4. **Push to branch**

   ```bash
   git push origin feature/new-feature-name
   ```

5. **Create Pull Request**

## License

This project is licensed under the [GNU GPLV3 License](LICENSE). For detailed information, please refer to the `LICENSE` file.

## Contact

For any questions or suggestions, please contact the project maintainers through [issues](https://github.com/bud-primordium/ThermoElasticSim/issues).