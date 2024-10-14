# tests/test_stress_evaluator.py

"""
@file test_stress_evaluator.py
@brief 测试 stress_evaluator.py 模块中的 StressEvaluator 类。
"""

import unittest
import numpy as np
from src.python.structure import CrystalStructure, Particle
from src.python.potentials import LennardJonesPotential
from src.python.stress_evaluator import LennardJonesStressEvaluator


class TestLennardJonesStressEvaluator(unittest.TestCase):
    """
    @class TestLennardJonesStressEvaluator
    @brief 测试 LennardJonesStressEvaluator 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        particles = [
            Particle(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0]),
            Particle(id=2, symbol="Al", mass=26.9815, position=[3.405, 0.0, 0.0]),
        ]
        lattice_vectors = [[3.405, 0.0, 0.0], [0.0, 3.405, 0.0], [0.0, 0.0, 3.405]]
        self.crystal = CrystalStructure(
            lattice_vectors=lattice_vectors, particles=particles
        )
        self.potential = LennardJonesPotential(
            parameters={"epsilon": 0.0103, "sigma": 3.405}, cutoff=5.0
        )
        self.stress_evaluator = LennardJonesStressEvaluator()

    def test_compute_stress(self) -> None:
        """
        @brief 测试应力计算函数是否返回正确的形状。
        """
        stress_voigt = self.stress_evaluator.compute_stress(
            self.crystal, self.potential
        )
        self.assertEqual(stress_voigt.shape, (6,))
        # 由于calculate_stress尚未实现，预期抛出NotImplementedError
        # 这里仅测试接口是否工作
        with self.assertRaises(NotImplementedError):
            _ = self.stress_evaluator.compute_stress(self.crystal, self.potential)


if __name__ == "__main__":
    unittest.main()
