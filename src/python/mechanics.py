# src/python/mechanics.py

import numpy as np
from .utils import TensorConverter


class StressCalculator:
    def compute_stress(self, cell, potential):
        raise NotImplementedError


class StressCalculatorLJ(StressCalculator):
    def compute_stress(self, cell, potential):
        volume = cell.calculate_volume()
        stress_tensor = np.zeros((3, 3))
        atoms = cell.atoms
        # Kinetic contribution
        for atom in atoms:
            m = atom.mass
            v = atom.velocity
            stress_tensor += m * np.outer(v, v)
        # Potential contribution
        for i, atom_i in enumerate(atoms):
            for atom_j in atoms[i + 1 :]:
                rij = atom_j.position - atom_i.position
                rij = cell.apply_periodic_boundary(rij)
                r = np.linalg.norm(rij)
                if r < potential.cutoff:
                    sr6 = (potential.sigma / r) ** 6
                    sr12 = sr6 * sr6
                    force_scalar = 24 * potential.epsilon * (2 * sr12 - sr6) / r
                    # Shifted force correction
                    force_scalar -= potential.dUc / r
                    fij = force_scalar * rij / r
                    stress_tensor += np.outer(rij, fij)
        stress_tensor /= volume
        return -stress_tensor  # Negative sign as per definition


class StrainCalculator:
    def compute_strain(self, deformation_gradient):
        C = np.dot(deformation_gradient.T, deformation_gradient)
        strain_tensor = 0.5 * (C - np.identity(3))
        return strain_tensor


class ElasticConstantsSolver:
    def solve(self, strains, stresses):
        strains = np.array(strains)
        stresses = np.array(stresses)
        C, residuals, rank, s = np.linalg.lstsq(strains, stresses, rcond=None)
        return C
