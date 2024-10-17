# src/python/potentials.py

import numpy as np


class Potential:
    def __init__(self, parameters, cutoff):
        self.parameters = parameters
        self.cutoff = cutoff

    def calculate_potential(self, cell):
        raise NotImplementedError

    def calculate_forces(self, cell):
        raise NotImplementedError


class LennardJonesPotential(Potential):
    def __init__(self, epsilon, sigma, cutoff):
        parameters = {"epsilon": epsilon, "sigma": sigma}
        super().__init__(parameters, cutoff)
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        # Pre-compute shifted potential and force at cutoff
        self.Uc = 4 * epsilon * ((sigma / cutoff) ** 12 - (sigma / cutoff) ** 6)
        self.dUc = (
            -48 * epsilon * ((sigma**12 / cutoff**13) - 0.5 * (sigma**6 / cutoff**7))
        )

    def calculate_potential(self, cell):
        energy = 0.0
        atoms = cell.atoms
        for i, atom_i in enumerate(atoms):
            for atom_j in atoms[i + 1 :]:
                rij = atom_j.position - atom_i.position
                rij = cell.apply_periodic_boundary(rij)
                r = np.linalg.norm(rij)
                if r < self.cutoff:
                    sr6 = (self.sigma / r) ** 6
                    sr12 = sr6 * sr6
                    U = 4 * self.epsilon * (sr12 - sr6)
                    # Shifted potential
                    U_shifted = U - self.Uc - self.dUc * (r - self.cutoff)
                    energy += U_shifted
        return energy

    def calculate_forces(self, cell):
        atoms = cell.atoms
        for atom in atoms:
            atom.force[:] = 0.0  # Reset forces
        for i, atom_i in enumerate(atoms):
            for atom_j in atoms[i + 1 :]:
                rij = atom_j.position - atom_i.position
                rij = cell.apply_periodic_boundary(rij)
                r = np.linalg.norm(rij)
                if r < self.cutoff:
                    sr6 = (self.sigma / r) ** 6
                    sr12 = sr6 * sr6
                    force_scalar = 24 * self.epsilon * (2 * sr12 - sr6) / r
                    # Shifted force correction
                    force_scalar -= self.dUc / r
                    force_vector = force_scalar * rij / r
                    atom_i.force += force_vector
                    atom_j.force -= force_vector
