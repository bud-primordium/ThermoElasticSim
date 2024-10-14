# tests/test_solver.py

"""
@file test_solver.py
@brief 测试 solver.py 模块中的 ElasticConstantSolver 类。
"""

import unittest
import numpy as np
from src.python.solver import ElasticConstantSolver


class TestElasticConstantSolver(unittest.TestCase):
    """
    @class TestElasticConstantSolver
    @brief 测试 ElasticConstantSolver 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        self.solver = ElasticConstantSolver()

    def test_solve(self) -> None:
        """
        @brief 测试弹性刚度矩阵求解是否正确。
        """
        # 示例应变和应力数据
        strain_data = [
            np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]),
        ]
        stress_data = [
            np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0]),
        ]

        C = self.solver.solve(strain_data, stress_data)

        # 预期弹性刚度矩阵为对角矩阵，主对角线为100 / 0.01 = 10000
        expected_C = np.diag([10000.0, 10000.0, 10000.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(C, expected_C)


if __name__ == "__main__":
    unittest.main()
