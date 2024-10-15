# tests/test_structure.py

"""
@file test_structure.py
@brief 测试 structure.py 模块中的 Atom 和 Cell 类。
"""

import unittest
import numpy as np
from src.python.structure import Atom, Cell


class TestAtom(unittest.TestCase):
    """
    @class TestAtom
    @brief 测试 Atom 类。
    """

    def test_atom_initialization(self) -> None:
        """
        @brief 测试原子初始化是否正确。
        """
        p = Atom(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0])
        self.assertEqual(p.id, 1)
        self.assertEqual(p.symbol, "Al")
        self.assertAlmostEqual(p.mass, 26.9815)
        np.testing.assert_array_equal(p.position, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(p.velocity, np.zeros(3))

    def test_update_position(self) -> None:
        """
        @brief 测试更新原子位置。
        """
        p = Atom(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0])
        p.update_position([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(p.position, np.array([1.0, 1.0, 1.0]))

    def test_update_velocity(self) -> None:
        """
        @brief 测试更新原子速度。
        """
        p = Atom(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0])
        p.update_velocity([0.5, 0.5, 0.5])
        np.testing.assert_array_equal(p.velocity, np.array([0.5, 0.5, 0.5]))


class TestCell(unittest.TestCase):
    """
    @class TestCell
    @brief 测试 Cell 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        self.atoms = [
            Atom(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0]),
            Atom(id=2, symbol="Al", mass=26.9815, position=[1.8075, 1.8075, 1.8075]),
        ]
        self.lattice_vectors = [[3.615, 0.0, 0.0], [0.0, 3.615, 0.0], [0.0, 0.0, 3.615]]
        self.cell = Cell(lattice_vectors=self.lattice_vectors, atoms=self.atoms)

    def test_volume_calculation(self) -> None:
        """
        @brief 测试晶胞体积计算。
        """
        expected_volume = 3.615**3
        self.assertAlmostEqual(self.cell.volume, expected_volume)

    def test_apply_deformation(self) -> None:
        """
        @brief 测试施加应变后的晶体结构更新。
        """
        F = np.array([[1.01, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.cell.apply_deformation(F)
        expected_lattice_vectors = np.dot(F, np.array(self.lattice_vectors))
        np.testing.assert_array_almost_equal(
            self.cell.lattice_vectors, expected_lattice_vectors
        )
        for i, atom in enumerate(self.cell.atoms):
            expected_position = np.dot(F, np.array(self.atoms[i].position))
            np.testing.assert_array_almost_equal(atom.position, expected_position)


if __name__ == "__main__":
    unittest.main()
