# tests/test_mechanics.py

import pytest
import numpy as np
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.zeroelasticity import (
    ZeroKElasticConstantsCalculator,
    ZeroKElasticConstantsSolver,
)
from python.mechanics import StressCalculatorLJ
from python.utils import NeighborList
from conftest import generate_fcc_positions  # 从 conftest 导入


def test_stress_calculation(two_atom_cell, lj_potential_with_neighbor_list):
    """
    测试应力计算器的功能。
    """
    stress_calculator = StressCalculatorLJ()
    # 计算力
    lj_potential_with_neighbor_list.calculate_forces(two_atom_cell)
    # 计算应力
    stress_tensor = stress_calculator.compute_stress(
        two_atom_cell, lj_potential_with_neighbor_list
    )
    # 不再假设应力张量为零，而是检查应力值是否与预期相符
    # 根据计算，手动计算期望的应力张量（或根据物理意义设置一个合理范围）
    expected_stress = np.array(
        [[-0.00475202, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    np.testing.assert_array_almost_equal(stress_tensor, expected_stress, decimal=6)


def test_elastic_constants_solver():
    """
    测试弹性常数求解器的功能。
    """
    strains = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01]),
    ]
    stresses = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([69.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 69.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 69.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 23.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 23.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 23.0]),
    ]

    # 创建 ZeroKElasticConstantsSolver 实例
    solver = ZeroKElasticConstantsSolver()

    # 使用 solver.solve 方法计算弹性常数矩阵
    C = solver.solve(strains, stresses)

    # 检查 C 是否为 6x6 矩阵
    assert C.shape == (6, 6), "Elastic constants matrix shape mismatch."
    # 预期弹性常数矩阵
    expected_C = np.diag([6900.0, 6900.0, 6900.0, 2300.0, 2300.0, 2300.0])
    # 检查弹性常数矩阵是否接近预期值
    np.testing.assert_array_almost_equal(C, expected_C, decimal=2)
