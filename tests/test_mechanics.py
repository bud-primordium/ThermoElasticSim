# tests/test_mechanics.py

import unittest
import numpy as np

from python.mechanics import (
    StressCalculatorLJ,
    StrainCalculator,
    ElasticConstantsSolver,
)
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential


class TestMechanics(unittest.TestCase):
    """
    @class TestMechanics
    @brief 测试力学计算模块的功能
    """

    def setUp(self):
        # 创建一个简单的晶胞
        lattice_vectors = np.eye(3) * 4.05  # Å
        mass = 26.9815  # 原子量，amu
        position = np.array([0.0, 0.0, 0.0])
        atom = Atom(id=0, symbol="Al", mass=mass, position=position)
        self.cell = Cell(lattice_vectors, [atom])

        # 定义 Lennard-Jones 势
        epsilon = 0.0103  # eV
        sigma = 2.55  # Å
        cutoff = 2.5 * sigma
        self.potential = LennardJonesPotential(
            epsilon=epsilon, sigma=sigma, cutoff=cutoff
        )

    def test_stress_calculation(self):
        """
        @brief 测试应力计算器的功能
        """
        stress_calculator = StressCalculatorLJ()
        self.potential.calculate_forces(self.cell)
        stress_tensor = stress_calculator.compute_stress(self.cell, self.potential)
        # 检查应力张量是否为 3x3 矩阵
        self.assertEqual(stress_tensor.shape, (3, 3))

    def test_strain_calculation(self):
        """
        @brief 测试应变计算器的功能
        """
        strain_calculator = StrainCalculator()
        F = np.array([[1.01, 0, 0], [0, 1, 0], [0, 0, 1]])
        strain_tensor = strain_calculator.compute_strain(F)
        # 检查应变张量是否为 3x3 矩阵
        self.assertEqual(strain_tensor.shape, (3, 3))

    def test_elastic_constants_solver(self):
        """
        @brief 测试弹性常数求解器的功能
        """
        strains = [np.zeros(6), np.ones(6) * 0.01]
        stresses = [np.zeros(6), np.ones(6)]
        solver = ElasticConstantsSolver()
        C = solver.solve(strains, stresses)
        # 检查 C 是否为 6x6 矩阵
        self.assertEqual(C.shape, (6, 6))


if __name__ == "__main__":
    unittest.main()
