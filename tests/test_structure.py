# tests/test_structure.py

import unittest
import numpy as np
import sys
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from python.structure import Atom, Cell


class TestStructure(unittest.TestCase):
    def test_atom_creation(self):
        position = np.array([0.0, 0.0, 0.0])
        mass = 26.9815 / (6.02214076e23) * 1e-3  # kg
        atom = Atom(id=0, symbol="Al", mass=mass, position=position)
        self.assertEqual(atom.id, 0)
        self.assertEqual(atom.symbol, "Al")
        np.testing.assert_array_equal(atom.position, position)
        self.assertEqual(atom.mass, mass)

    def test_cell_creation(self):
        lattice_vectors = np.eye(3) * 4.05e-10  # 米
        mass = 26.9815 / (6.02214076e23) * 1e-3
        atom = Atom(id=0, symbol="Al", mass=mass, position=np.zeros(3))
        cell = Cell(lattice_vectors, [atom])
        np.testing.assert_array_equal(cell.lattice_vectors, lattice_vectors)
        self.assertEqual(len(cell.atoms), 1)
        self.assertEqual(cell.atoms[0], atom)

    def test_volume_calculation(self):
        lattice_vectors = np.eye(3) * 4.05e-10  # 米
        cell = Cell(lattice_vectors, [])
        expected_volume = (4.05e-10) ** 3
        calculated_volume = cell.calculate_volume()
        self.assertAlmostEqual(calculated_volume, expected_volume, places=20)


if __name__ == "__main__":
    unittest.main()
