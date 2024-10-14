# tests/test_strain.py

"""
@file test_strain.py
@brief 测试 strain.py 模块中的 StrainCalculator 类。
"""

import unittest
import numpy as np
from src.python.strain import StrainCalculator


class TestStrainCalculator(unittest.TestCase):
    """
    @class TestStrainCalculator
    @brief 测试 StrainCalculator 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        self.calculator = StrainCalculator()

    def test_calculate_strain(self) -> None:
        """
        @brief 测试应变张量的计算是否正确。
        """
        F = np.array([[1.01, 0.0, 0.0], [0.0, 1.02, 0.0], [0.0, 0.0, 1.03]])
        expected_E = 0.5 * (np.dot(F.T, F) - np.identity(3))
        expected_strain_voigt = np.array(
            [
                expected_E[0, 0],
                expected_E[1, 1],
                expected_E[2, 2],
                2 * expected_E[1, 2],
                2 * expected_E[0, 2],
                2 * expected_E[0, 1],
            ]
        )
        calculated_strain_voigt = self.calculator.calculate_strain(F)
        np.testing.assert_array_almost_equal(
            calculated_strain_voigt, expected_strain_voigt
        )


if __name__ == "__main__":
    unittest.main()
