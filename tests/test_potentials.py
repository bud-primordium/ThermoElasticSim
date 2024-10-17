# tests/test_potentials.py

import unittest
import numpy as np

from python.potentials import LennardJonesPotential
from python.structure import Atom, Cell


class TestPotentials(unittest.TestCase):
    def setUp(self):
        self.epsilon = 6.774e-21  # J
        self.sigma = 2.55e-10  # m
        self.cutoff = 2.5 * self.sigma
        self.potential = LennardJonesPotential(
            epsilon=self.epsilon, sigma=self.sigma, cutoff=self.cutoff
        )
        # 创建一个简单的系统
        self.lattice_vectors = np.eye(3) * 4.05e-10
        mass = 26.9815 / (6.02214076e23) * 1e-3
        position1 = np.array([0.0, 0.0, 0.0])
        position2 = np.array([2.025e-10, 0.0, 0.0])
        atom1 = Atom(id=0, symbol="Al", mass=mass, position=position1)
        atom2 = Atom(id=1, symbol="Al", mass=mass, position=position2)
        self.cell = Cell(self.lattice_vectors, [atom1, atom2])

    def test_force_calculation(self):
        self.potential.calculate_forces(self.cell)
        # 检查力是否非零且相反
        force1 = self.cell.atoms[0].force
        force2 = self.cell.atoms[1].force
        np.testing.assert_array_almost_equal(force1, -force2, decimal=10)
        self.assertFalse(np.allclose(force1, 0))


if __name__ == "__main__":
    unittest.main()
