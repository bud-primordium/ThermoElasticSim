# tests/test_deformation.py

"""
@file test_deformation.py
@brief 测试 deformation.py 模块中的 Deformer 类。
"""

import unittest
import numpy as np
from src.python.deformation import Deformer
from src.python.structure import CrystalStructure, Atom


class TestDeformer(unittest.TestCase):
    """
    @class TestDeformer
    @brief 测试 Deformer 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        atoms = [
            Atom(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0]),
            Atom(id=2, symbol="Al", mass=26.9815, position=[1.8075, 1.8075, 1.8075]),
            # 添加更多原子
        ]
        lattice_vectors = [[3.615, 0.0, 0.0], [0.0, 3.615, 0.0], [0.0, 0.0, 3.615]]
        self.crystal = CrystalStructure(lattice_vectors=lattice_vectors, atoms=atoms)
        self.deformer = Deformer()

    def test_generate_deformations(self) -> None:
        """
        @brief 测试应变矩阵的生成是否正确。
        """
        delta = 0.01
        strains = self.deformer.generate_deformations(delta=delta)
        self.assertEqual(len(strains), 6)
        for F in strains:
            self.assertEqual(F.shape, (3, 3))
            self.assertTrue(
                np.allclose(F, np.identity(3) + F - np.identity(3), atol=1e-8)
            )  # 检查是否接近应变

    def test_apply_deformation(self) -> None:
        """
        @brief 测试应变矩阵的应用是否正确更新晶体结构。
        """
        F = np.array([[1.01, 0.0, 0.0], [0.0, 1.02, 0.0], [0.0, 0.0, 1.03]])
        deformed_crystal = self.deformer.apply_deformation(self.crystal, F)

        expected_lattice_vectors = np.dot(F, np.array(self.crystal.lattice_vectors))
        np.testing.assert_array_almost_equal(
            deformed_crystal.lattice_vectors, expected_lattice_vectors
        )

        for i, atom in enumerate(deformed_crystal.atoms):
            expected_position = np.dot(F, np.array(self.crystal.atoms[i].position))
            np.testing.assert_array_almost_equal(atom.position, expected_position)


if __name__ == "__main__":
    unittest.main()
