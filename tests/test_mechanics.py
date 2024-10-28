# tests/test_mechanics.py

import pytest
import numpy as np
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.zeroelasticity import (
    ZeroKElasticConstantsCalculator,
    ZeroKElasticConstantsSolver,
)
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


def test_zero_elastic_constants():
    """
    测试零温度下的弹性常数计算，确保在变形后进行结构优化。
    """
    lattice_constant = 4.05  # Å
    repetitions = 2

    # 生成 2x2x2 超胞的原子位置
    positions = generate_fcc_positions(lattice_constant, repetitions)
    atoms = []
    for idx, pos in enumerate(positions):
        atoms.append(
            Atom(
                id=idx,
                symbol="Al",
                mass_amu=26.9815,
                position=pos,
            )
        )

    # 更新晶格矢量
    lattice_vectors = np.eye(3) * lattice_constant * repetitions

    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 定义 Lennard-Jones 势能
    lj_potential = LennardJonesPotential(epsilon=0.0103, sigma=2.55, cutoff=2.5 * 2.55)

    # 创建邻居列表并关联到势能函数
    neighbor_list = NeighborList(cutoff=lj_potential.cutoff)
    neighbor_list.build(cell)
    lj_potential.set_neighbor_list(neighbor_list)

    # 创建 ZeroKElasticConstantsCalculator 实例
    elastic_calculator = ZeroKElasticConstantsCalculator(
        cell=cell,
        potential=lj_potential,
        delta=1e-3,
        optimizer_type="GD",  # 使用梯度下降优化器
    )

    # 计算弹性常数
    C_in_GPa = elastic_calculator.calculate_elastic_constants()

    # 检查 C_in_GPa 是一个 6x6 的矩阵
    assert C_in_GPa.shape == (6, 6), "弹性常数矩阵形状不匹配。"

    # 检查对称性
    assert np.allclose(C_in_GPa, C_in_GPa.T, atol=1e-3), "弹性常数矩阵不是对称的。"

    # 检查对角元素为正
    for i in range(6):
        assert C_in_GPa[i, i] > 0, f"弹性常数 C[{i},{i}] 不是正值。"

    # 检查非对角元素在合理范围内
    for i in range(6):
        for j in range(i + 1, 6):
            assert (
                -100.0 <= C_in_GPa[i, j] <= 100.0
            ), f"弹性常数 C[{i},{j}] 不在合理范围内。"
