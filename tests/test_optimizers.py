# tests/test_optimizers.py

"""
@file test_optimizers.py
@brief 测试 optimizers.py 模块中的结构优化算法。
"""

import unittest
import numpy as np
from src.python.structure import CrystalStructure, Atom
from src.python.potentials import LennardJonesPotential
from src.python.optimizers import ConjugateGradientOptimizer, NewtonRaphsonOptimizer


class TestConjugateGradientOptimizer(unittest.TestCase):
    """
    @class TestConjugateGradientOptimizer
    @brief 测试 ConjugateGradientOptimizer 类。
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
        self.potential = LennardJonesPotential(
            parameters={"epsilon": 0.0103, "sigma": 3.405}, cutoff=5.0
        )
        self.optimizer = ConjugateGradientOptimizer()

    def test_optimize_not_implemented(self) -> None:
        """
        @brief 测试 ConjugateGradientOptimizer.optimize 方法是否抛出 NotImplementedError。
        """
        with self.assertRaises(NotImplementedError):
            self.optimizer.optimize(self.crystal, self.potential)


class TestNewtonRaphsonOptimizer(unittest.TestCase):
    """
    @class TestNewtonRaphsonOptimizer
    @brief 测试 NewtonRaphsonOptimizer 类。
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
        self.potential = LennardJonesPotential(
            parameters={"epsilon": 0.0103, "sigma": 3.405}, cutoff=5.0
        )
        self.optimizer = NewtonRaphsonOptimizer()

    def test_optimize_not_implemented(self) -> None:
        """
        @brief 测试 NewtonRaphsonOptimizer.optimize 方法是否抛出 NotImplementedError。
        """
        with self.assertRaises(NotImplementedError):
            self.optimizer.optimize(self.crystal, self.potential)


if __name__ == "__main__":
    unittest.main()
