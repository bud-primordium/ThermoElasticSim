# tests/test_optimizers.py

import unittest
import numpy as np
import sys
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from python.optimizers import GradientDescentOptimizer
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的系统（两个铝原子）
        lattice_vectors = np.eye(3) * 4.05e-10  # 米
        mass = 26.9815 / (6.02214076e23) * 1e-3  # kg，正确的质量单位
        position1 = np.array([0.0, 0.0, 0.0])
        position2 = np.array([2.55e-10, 0.0, 0.0])  # 初始距离为 sigma
        atom1 = Atom(id=0, symbol="Al", mass=mass, position=position1)
        atom2 = Atom(id=1, symbol="Al", mass=mass, position=position2)
        self.cell = Cell(
            lattice_vectors, [atom1, atom2], pbc_enabled=False
        )  # 禁用周期性边界条件

        # 定义 Lennard-Jones 势
        epsilon = 6.774e-21  # J
        sigma = 2.55e-10  # m
        cutoff = 2.5 * sigma
        self.potential = LennardJonesPotential(
            epsilon=epsilon, sigma=sigma, cutoff=cutoff
        )

    def test_gradient_descent_optimizer(self):
        optimizer = GradientDescentOptimizer(
            max_steps=1000, tol=1e-8, step_size=1e-12  # 使用适当的步长
        )
        optimizer.optimize(self.cell, self.potential)

        # 获取优化后的原子位置
        optimized_position1 = self.cell.atoms[0].position
        optimized_position2 = self.cell.atoms[1].position

        # 计算优化后的距离
        optimized_distance = np.linalg.norm(optimized_position2 - optimized_position1)

        # 预期优化后的距离应接近 2^(1/6) * sigma ≈ 2.86e-10 m
        equilibrium_distance = 2 ** (1 / 6) * self.potential.sigma
        self.assertAlmostEqual(optimized_distance, equilibrium_distance, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
