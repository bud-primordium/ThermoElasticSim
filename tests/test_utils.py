# tests/test_structure.py

import unittest
import numpy as np

from python.structure import Atom, Cell


class TestStructure(unittest.TestCase):
    """
    @class TestStructure
    @brief 测试原子和晶胞结构的功能
    """

    def test_atom_creation(self):
        """
        @brief 测试原子的创建
        """
        position = np.array([0.0, 0.0, 0.0])
        mass = 26.9815  # 原子量，amu
        atom = Atom(id=0, symbol="Al", mass=mass, position=position)
        self.assertEqual(atom.id, 0)
        self.assertEqual(atom.symbol, "Al")
        np.testing.assert_array_equal(atom.position, position)
        self.assertEqual(atom.mass, mass)

    def test_cell_creation(self):
        """
        @brief 测试晶胞的创建
        """
        lattice_vectors = np.eye(3) * 4.05  # Å
        atom = Atom(id=0, symbol="Al", mass=26.9815, position=np.zeros(3))
        cell = Cell(lattice_vectors, [atom])
        np.testing.assert_array_equal(cell.lattice_vectors, lattice_vectors)
        self.assertEqual(len(cell.atoms), 1)
        self.assertEqual(cell.atoms[0], atom)

    def test_volume_calculation(self):
        """
        @brief 测试晶胞体积的计算
        """
        lattice_vectors = np.eye(3) * 4.05  # Å
        cell = Cell(lattice_vectors, [])
        expected_volume = 4.05**3
        calculated_volume = cell.calculate_volume()
        self.assertAlmostEqual(calculated_volume, expected_volume)

    def test_apply_periodic_boundary(self):
        """
        @brief 测试周期性边界条件的应用
        """
        lattice_vectors = np.eye(3) * 4.05  # Å
        cell = Cell(lattice_vectors, [], pbc_enabled=True)
        position = np.array([5.0, -1.0, 4.0])  # 超出晶胞范围的坐标
        new_position = cell.apply_periodic_boundary(position)
        # 检查新位置是否在 [0, lattice_constant) 范围内
        self.assertTrue(np.all(new_position >= 0))
        self.assertTrue(np.all(new_position < 4.05))


if __name__ == "__main__":
    unittest.main()
