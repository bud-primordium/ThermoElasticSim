# src/python/optimizers.py

import numpy as np


class Optimizer:
    def optimize(self, cell, potential):
        raise NotImplementedError


class QuickminOptimizer(Optimizer):
    def __init__(self, max_steps=1000, tol=1e-6, dt=1e-3):
        self.max_steps = max_steps
        self.tol = tol
        self.dt = dt

    def optimize(self, cell, potential):
        atoms = cell.atoms
        velocities = [np.zeros(3) for _ in atoms]
        potential.calculate_forces(cell)
        for step in range(self.max_steps):
            max_force = max(np.linalg.norm(atom.force) for atom in atoms)
            if max_force < self.tol:
                print(f"Converged after {step} steps")
                break
            # Update velocities and positions
            for i, atom in enumerate(atoms):
                velocities[i] = velocities[i] + self.dt * atom.force / atom.mass
                atom.position += self.dt * velocities[i]
                # Apply periodic boundary conditions
                atom.position = cell.apply_periodic_boundary(atom.position)
            # Calculate new forces
            potential.calculate_forces(cell)
        else:
            print("Optimization did not converge within the maximum number of steps.")
