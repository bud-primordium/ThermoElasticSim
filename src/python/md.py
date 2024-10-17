# src/python/md.py

import numpy as np
from .interfaces.fortran_interface import FortranInterface


class Integrator:
    def integrate(self, cell, potential, thermostat, dt):
        raise NotImplementedError


class VelocityVerletIntegrator(Integrator):
    def integrate(self, cell, potential, thermostat, dt):
        atoms = cell.atoms
        # First half-step: update positions
        for atom in atoms:
            atom.position += atom.velocity * dt + 0.5 * atom.force / atom.mass * dt**2
            # Apply periodic boundary conditions
            atom.position = cell.apply_periodic_boundary(atom.position)
        # Save old forces
        forces_old = [atom.force.copy() for atom in atoms]
        # Calculate new forces
        potential.calculate_forces(cell)
        # Second half-step: update velocities
        for atom, force_old in zip(atoms, forces_old):
            atom.velocity += 0.5 * (atom.force + force_old) / atom.mass * dt
        # Apply thermostat
        if thermostat is not None:
            thermostat.apply(atoms, dt)


class Thermostat:
    def apply(self, atoms, dt):
        raise NotImplementedError


class NoseHooverThermostat(Thermostat):
    def __init__(self, target_temperature, time_constant, num_chains=3):
        self.target_temperature = target_temperature
        self.time_constant = time_constant  # Q parameter
        self.num_chains = num_chains
        self.xi = np.zeros(num_chains)
        self.eta = np.zeros(num_chains)
        self.fortran_interface = FortranInterface("path/to/fortran/library")

    def apply(self, atoms, dt):
        num_atoms = len(atoms)
        masses = np.array([atom.mass for atom in atoms], dtype=np.float64)
        positions = np.array(
            [atom.position for atom in atoms], dtype=np.float64
        ).T  # Transposed for Fortran
        velocities = np.array([atom.velocity for atom in atoms], dtype=np.float64).T
        forces = np.array([atom.force for atom in atoms], dtype=np.float64).T
        temperature = self.target_temperature
        Q = self.time_constant

        # Call Fortran subroutine
        self.fortran_interface.nose_hoover_chain(
            dt,
            num_atoms,
            masses,
            positions,
            velocities,
            forces,
            Q,
            self.xi,
            temperature,
        )

        # Update atom velocities
        for i, atom in enumerate(atoms):
            atom.velocity = velocities[:, i]


class MDSimulator:
    def __init__(self, cell, potential, integrator, thermostat=None):
        self.cell = cell
        self.potential = potential
        self.integrator = integrator
        self.thermostat = thermostat

    def run(self, steps, dt, data_collector=None):
        # Initialize forces
        self.potential.calculate_forces(self.cell)
        for step in range(steps):
            self.integrator.integrate(self.cell, self.potential, self.thermostat, dt)
            if data_collector is not None:
                data_collector.collect(self.cell)
