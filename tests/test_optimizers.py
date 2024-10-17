# tests/test_optimizers.py

import unittest
import numpy as np

from python.optimizers import GradientDescentOptimizer, QuickminOptimizer
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential


class TestOptimizers(unittest.TestCase):
    """
    @class TestOptimizers
    @brief 测试优化器的功能
    """

    def setUp(self):
        # 创建一个简单的晶胞，包含两个原子
        lattice_vectors = np.eye(3) * 4.05  # Å
        mass = 26.9815  # 原子量，amu
        position1 = np.array([0.0, 0.0, 0.0])
        position2 = np.array([2.55, 0.0, 0.0])  # 初始距离为 σ
        atom1 = Atom(id=0, symbol="Al", mass=mass, position=position1)
        atom2 = Atom(id=1, symbol="Al", mass=mass, position=position2)
        self.cell = Cell(lattice_vectors, [atom1, atom2], pbc_enabled=False)

        # 定义 Lennard-Jones 势
        epsilon = 0.0103  # eV
        sigma = 2.55  # Å
        cutoff = 2.5 * sigma
        self.potential = LennardJonesPotential(
            epsilon=epsilon, sigma=sigma, cutoff=cutoff
        )

    def test_gradient_descent_optimizer(self):
        """
        @brief 测试梯度下降优化器
        """
        optimizer = GradientDescentOptimizer(max_steps=1000, tol=1e-6, step_size=1e-4)
        optimizer.optimize(self.cell, self.potential)

        # 获取优化后的原子位置
        optimized_position1 = self.cell.atoms[0].position
        optimized_position2 = self.cell.atoms[1].position

        # 计算优化后的距离
        optimized_distance = np.linalg.norm(optimized_position2 - optimized_position1)

        # 预期优化后的距离应接近 2^(1/6) * σ ≈ 2.857 Å
        equilibrium_distance = 2 ** (1 / 6) * self.potential.sigma
        self.assertAlmostEqual(optimized_distance, equilibrium_distance, places=3)

    def test_quickmin_optimizer(self):
        """
        @brief 测试 Quickmin 优化器
        """
        optimizer = QuickminOptimizer(max_steps=1000, tol=1e-6, dt=0.01)
        optimizer.optimize(self.cell, self.potential)

        # 获取优化后的原子位置
        optimized_position1 = self.cell.atoms[0].position
        optimized_position2 = self.cell.atoms[1].position

        # 计算优化后的距离
        optimized_distance = np.linalg.norm(optimized_position2 - optimized_position1)

        # 预期优化后的距离应接近 2^(1/6) * σ ≈ 2.857 Å
        equilibrium_distance = 2 ** (1 / 6) * self.potential.sigma
        self.assertAlmostEqual(optimized_distance, equilibrium_distance, places=3)


if __name__ == "__main__":
    unittest.main()
