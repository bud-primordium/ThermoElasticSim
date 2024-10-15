# tests/test_visualization.py

"""
@file test_visualization.py
@brief 测试 visualization.py 模块中的 Visualizer 类。
"""

import unittest
from src.python.visualization import Visualizer
from src.python.structure import CrystalStructure, Particle
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 使用非GUI后端以避免在测试环境中弹出窗口


class TestVisualizer(unittest.TestCase):
    """
    @class TestVisualizer
    @brief 测试 Visualizer 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        particles = [
            Particle(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0]),
            Particle(
                id=2, symbol="Al", mass=26.9815, position=[1.8075, 1.8075, 1.8075]
            ),
        ]
        lattice_vectors = [[3.405, 0.0, 0.0], [0.0, 3.405, 0.0], [0.0, 0.0, 3.405]]
        self.crystal = CrystalStructure(
            lattice_vectors=lattice_vectors, particles=particles
        )
        self.visualizer = Visualizer()

    def test_plot_crystal_structure(self) -> None:
        """
        @brief 测试绘制晶体结构是否成功（不抛出异常）。
        """
        try:
            self.visualizer.plot_crystal_structure(self.crystal)
        except Exception as e:
            self.fail(f"plot_crystal_structure 抛出异常: {e}")

    def test_plot_stress_strain(self) -> None:
        """
        @brief 测试绘制应力-应变关系是否成功（不抛出异常）。
        """
        strain_data = np.array(
            [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 0.0, 0.0, 0.0]]
        )
        stress_data = np.array(
            [[100.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]]
        )
        try:
            self.visualizer.plot_stress_strain(strain_data, stress_data)
        except Exception as e:
            self.fail(f"plot_stress_strain 抛出异常: {e}")


if __name__ == "__main__":
    unittest.main()