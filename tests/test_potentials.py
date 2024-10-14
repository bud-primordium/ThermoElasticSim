# tests/test_potentials.py

"""
@file test_potentials.py
@brief 测试 potentials.py 模块中的 Potential 类及其子类。
"""

import unittest
from src.python.potentials import LennardJonesPotential, EAMPotential


class TestLennardJonesPotential(unittest.TestCase):
    """
    @class TestLennardJonesPotential
    @brief 测试 LennardJonesPotential 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        self.parameters = {"epsilon": 0.0103, "sigma": 3.405}
        self.cutoff = 5.0
        self.potential = LennardJonesPotential(
            parameters=self.parameters, cutoff=self.cutoff
        )

    def test_calculate_potential(self) -> None:
        """
        @brief 测试 Lennard-Jones 势能的计算。
        """
        r = 3.405  # sigma
        expected_potential = 4 * self.parameters["epsilon"] * ((1) ** 12 - (1) ** 6)
        calculated_potential = self.potential.calculate_potential(r)
        self.assertAlmostEqual(calculated_potential, expected_potential)

    def test_derivative_potential(self) -> None:
        """
        @brief 测试 Lennard-Jones 势能导数的计算。
        """
        r = 3.405  # sigma
        expected_derivative = 24 * self.parameters["epsilon"] * (2 * 1**12 - 1**6) / r
        calculated_derivative = self.potential.derivative_potential(r)
        self.assertAlmostEqual(calculated_derivative, expected_derivative)


class TestEAMPotential(unittest.TestCase):
    """
    @class TestEAMPotential
    @brief 测试 EAMPotential 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        self.parameters = {"some_eam_param": 1.0}  # 示例参数
        self.cutoff = 5.0
        self.potential = EAMPotential(parameters=self.parameters, cutoff=self.cutoff)

    def test_calculate_potential_not_implemented(self) -> None:
        """
        @brief 测试 EAMPotential.calculate_potential 方法是否抛出 NotImplementedError。
        """
        with self.assertRaises(NotImplementedError):
            self.potential.calculate_potential(3.405)

    def test_derivative_potential_not_implemented(self) -> None:
        """
        @brief 测试 EAMPotential.derivative_potential 方法是否抛出 NotImplementedError。
        """
        with self.assertRaises(NotImplementedError):
            self.potential.derivative_potential(3.405)


if __name__ == "__main__":
    unittest.main()
