# examples/calculate_elastic_constants.py

import numpy as np
from src.python.structure import Atom, Cell
from src.python.potentials import LennardJonesPotential
from src.python.optimizers import QuickminOptimizer
from src.python.deformation import Deformer
from src.python.md import MDSimulator, VelocityVerletIntegrator, NoseHooverThermostat
from src.python.mechanics import (
    StressCalculatorLJ,
    StrainCalculator,
    ElasticConstantsSolver,
)
from src.python.utils import TensorConverter

# Define FCC lattice vectors
lattice_constant = 5.260  # Example value in angstroms
lattice_vectors = 0.5 * lattice_constant * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Define atom positions in fractional coordinates
fractional_positions = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]

atoms = []
for i, frac_pos in enumerate(fractional_positions):
    position = np.dot(frac_pos, lattice_vectors)
    atom = Atom(id=i, symbol="Ar", mass=39.948, position=position)
    atoms.append(atom)

cell = Cell(lattice_vectors, atoms)

# Define Lennard-Jones potential parameters for Argon
epsilon = 1.654e-21  # Joules
sigma = 3.4e-10  # Meters
cutoff = 2.5 * sigma

potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

# Structure optimization
optimizer = QuickminOptimizer(max_steps=500, tol=1e-8)
optimizer.optimize(cell, potential)

# Generate deformation matrices
delta = 0.01  # Deformation magnitude
deformer = Deformer(delta)
deformation_matrices = deformer.generate_deformation_matrices()

# Simulation parameters
temperatures = [100, 200, 300]  # Different temperatures in Kelvin

for T in temperatures:
    print(f"Calculating elastic constants at {T} K")
    strains = []
    stresses = []
    for F in deformation_matrices:
        # Create a copy of the cell and apply deformation
        deformed_cell = cell.copy()
        deformer.apply_deformation(deformed_cell, F)

        # Molecular dynamics simulation
        integrator = VelocityVerletIntegrator()
        thermostat = NoseHooverThermostat(target_temperature=T, time_constant=100)
        md_simulator = MDSimulator(deformed_cell, potential, integrator, thermostat)
        md_simulator.run(steps=1000, dt=1e-15)

        # Compute stress and strain
        stress_calculator = StressCalculatorLJ()
        stress_tensor = stress_calculator.compute_stress(deformed_cell, potential)
        strain_calculator = StrainCalculator()
        strain_tensor = strain_calculator.compute_strain(F)

        # Convert to Voigt notation
        stress_voigt = TensorConverter.to_voigt(stress_tensor)
        strain_voigt = TensorConverter.to_voigt(strain_tensor)

        strains.append(strain_voigt)
        stresses.append(stress_voigt)

    # Solve for elastic constants
    solver = ElasticConstantsSolver()
    C = solver.solve(strains, stresses)
    print(f"Elastic constants at {T} K:")
    print(C)
