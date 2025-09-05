# ThermoElasticSim

**Languages**: [中文](README.md) | English

**Documentation (in progress)**: https://bud-primordium.github.io/ThermoElasticSim/

## Project Overview

**ThermoElasticSim** is a molecular dynamics simulation tool specifically designed for calculating elastic constants of aluminum (Al) at zero and finite temperatures. Based on the EAM (Embedded Atom Method) potential, this project combines Python's ease of use with C++'s high-performance computing capabilities, employing explicit deformation and stress fluctuation methods to accurately calculate elastic properties of materials through structural optimization and molecular dynamics (MD) simulations.

The project implements multiple MD ensembles using an operator splitting architecture, including NVE, NVT (Berendsen, Andersen, Nosé-Hoover chain), and NPT (MTK), providing a reliable statistical mechanics foundation for finite-temperature elastic constant calculations.

## Key Features

### Implemented Features

1. **Zero-Temperature Elastic Constants (FCC metals + Diamond)**
   - Support for Aluminum (Al) and Copper (Cu) materials
   - Support for Diamond (C, diamond, Tersoff 1988)
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

- Elastic wave propagation simulation
- More teaching scenarios and visualization bindings (trajectory/energy/stress)

## Potential Models and Benchmark Accuracy

The project employs validated EAM (Embedded Atom Method) potentials for FCC metals molecular dynamics simulations:

### Aluminum (Al) Potential

**EAM_Dynamo_MendelevKramerBecker_2008_Al__MO_106969701023_006**

- **OpenKIM Database**: [MO_106969701023_006](https://openkim.org/id/EAM_Dynamo_MendelevKramerBecker_2008_Al__MO_106969701023_006)
- **Theoretical Basis**: Mendelev MI, Kramer MJ, Becker CA, Asta M. *Analysis of semi-empirical interatomic potentials appropriate for simulation of crystalline and liquid Al and Cu*. Philosophical Magazine. 2008;88(12):1723–50. [doi:10.1080/14786430802206482](https://doi.org/10.1080/14786430802206482)
- **Applicable Range**: Structural and dynamic properties of crystalline and liquid aluminum
- **Validation Accuracy**: Zero-temperature elastic constants error < 1.3% compared to literature values, good agreement with experimental values at finite temperatures

### Copper (Cu) Potential

**EAM_Dynamo_MendelevKramerBecker_2008_Cu__MO_945691923444_006**

- **OpenKIM Database**: [MO_945691923444_006](https://openkim.org/id/EAM_Dynamo_MendelevKramerBecker_2008_Cu__MO_945691923444_006)
- **Theoretical Basis**: Same as above, Mendelev et al. 2008
- **Applicable Range**: Structural and dynamic properties of crystalline and liquid copper
- **Validation Accuracy**: Zero-temperature C44 error 0.23%, uniaxial elastic constants slightly high, under further refinement

### Carbon (diamond) Potential

**Tersoff_LAMMPS_Tersoff_1988_C__MO_579868029681_004**

- **OpenKIM Database**: [MO_579868029681_004](https://openkim.org/id/Tersoff_LAMMPS_Tersoff_1988_C__MO_579868029681_004)
- **Theoretical Basis**: Tersoff J. *Empirical Interatomic Potential for Carbon, with Applications to Amorphous Carbon*. Phys. Rev. Lett. 61, 2879 (1988). [doi:10.1103/PhysRevLett.61.2879](https://doi.org/10.1103/PhysRevLett.61.2879)
- **Applicable Range (Abstract)**: Empirical potential accurately capturing carbon's structural and energetic features (including elasticity, phonons, polytypes, defects, and migration barriers in diamond/graphite), applied to amorphous carbon formation via multiple routes.
- **Model Features**: C++ backend analytical implementation (energy/forces/three-body virial); triplet-cluster virial accounting; tension-positive stress convention.
- **Zero-Temperature Equilibrium Lattice Constant**: a0 = 3.5656 Å (consistent with reference)
- **Validation Accuracy (vs OpenKIM reference)**:
  - Uniaxial (C11, C12): ≤ 0.02% error
  - Shear (C44): ≈ 4.6% error

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
- **[uv](https://github.com/astral-sh/uv)** (recommended) or pip

### Install with [uv](https://github.com/astral-sh/uv) (recommended)

```bash
# 0) Install uv (pick one)
# macOS (Homebrew)
brew install uv
# macOS/Linux (generic script)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 1) Clone repository
git clone https://github.com/bud-primordium/ThermoElasticSim.git
cd ThermoElasticSim

# 2) Create and activate a virtual environment
uv venv && source .venv/bin/activate

# 3) Install the project in editable mode (builds C++ extension)
uv pip install -e .

# 4) Verify the C++ extension
python -c "import thermoelasticsim._cpp_core as m; print('✓', m.__file__)"
```

### Running Tests

```bash
# Developers: install test dependencies first
uv pip install -e ".[dev]"

# Run tests (prefer module form to avoid PATH issues)
python -m pytest
# Or: uv run --no-sync python -m pytest

# Optional: if you insist on calling pytest directly, ensure venv binary
.venv/bin/pytest
```

Tip: If Conda base is active, the `pytest` command may resolve to the base
environment, bypassing your venv. Using `python -m pytest` guarantees the
venv's interpreter and dependencies are used.

### Installation Notes

- Automated build: `scikit-build-core` + `pybind11`.
- Editable mode: `redirect` – the `.so` lives in `site-packages/thermoelasticsim/`, sources import from `src/`, build artifacts go to `build/{wheel_tag}`.
- uv tip: Prefer venv's `python/pytest`; if you use `uv run` for tests, add `--no-sync`.

### FAQ

- ImportError: “pybind11 module _cpp_core is not available …”
  - Cause: The C++ extension has not been built/installed yet, or the editable mapping was replaced during `uv run`'s dependency sync.
  - Fix: `uv pip install -e ".[dev]"` then use `pytest` (or `uv run --no-sync pytest`).

- NumPy import failure (missing C extensions)
  - The project dependencies are upgraded to the new stack: `numpy>=2.0`, `numba>=0.60`, `scipy>=1.11`.
  - If your mirror provides sdists and you end up without wheels, force wheel install:
    `uv pip install -U --only-binary=:all: "numpy>=2" "numba>=0.60" "scipy>=1.11"`.

## Usage

### YAML scenarios (recommended)

Drive the program with a single YAML file; outputs match benchmark style:

```bash
# Zero-temperature (multi-size sweep)
python -m thermoelasticsim.cli.run -c examples/modern_yaml/zero_temp_elastic.yaml

# Finite-temperature (preheat → NPT → NHC production)
python -m thermoelasticsim.cli.run -c examples/modern_yaml/finite_temp_elastic.yaml

# Teaching units
python -m thermoelasticsim.cli.run -c examples/modern_yaml/relax.yaml          # zero-T relaxation (E–V/E–s curves)
python -m thermoelasticsim.cli.run -c examples/modern_yaml/nve.yaml            # NVE (temperature/energy)
python -m thermoelasticsim.cli.run -c examples/modern_yaml/nvt_langevin.yaml   # NVT-Langevin
python -m thermoelasticsim.cli.run -c examples/modern_yaml/nvt_nhc.yaml        # NVT-NHC
python -m thermoelasticsim.cli.run -c examples/modern_yaml/nvt_andersen.yaml   # NVT-Andersen
python -m thermoelasticsim.cli.run -c examples/modern_yaml/nvt_berendsen.yaml  # NVT-Berendsen
python -m thermoelasticsim.cli.run -c examples/modern_yaml/npt.yaml            # NPT-MTK
```

Each YAML is documented (Chinese comments), including material/structure, potential, timestep/steps/sampling, thermostat/barostat parameters, and zero-T strain options.

### Legacy Python examples (kept for reference)

```bash
uv run python examples/legacy_py/zero_temp_al_benchmark.py
uv run python examples/legacy_py/finite_temp_al_benchmark.py
```

## Configuration

All YAML examples live under `examples/modern_yaml/`, covering full workflows and modular teaching scenarios.
Each YAML contains human-friendly comments for quick adaptation.

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
