# tests/test_mechanics.py

import pytest
import numpy as np
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.elasticity import ElasticConstantsSolver  # 确保导入
from python.mechanics import StressCalculatorLJ


@pytest.fixture
def single_atom_cell():
    atoms = [Atom(id=0, mass=26.9815, position=np.array([0.0, 0.0, 0.0]), symbol="Al")]
    lattice_vectors = np.eye(3) * 5.1
    return Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)


@pytest.fixture
def lj_potential_single():
    return LennardJonesPotential(epsilon=0.0103, sigma=2.55, cutoff=2.5 * 2.55)


def test_stress_calculation(single_atom_cell, lj_potential_single):
    """
    @brief 测试应力计算器的功能。
    """
    stress_calculator = StressCalculatorLJ()
    # 计算力
    lj_potential_single.calculate_forces(single_atom_cell)
    # 计算应力
    stress_tensor = stress_calculator.compute_stress(
        single_atom_cell, lj_potential_single
    )
    # 由于只有一个原子，理论上应力张量应为零
    expected_stress = np.zeros((3, 3))
    np.testing.assert_array_almost_equal(stress_tensor, expected_stress, decimal=6)


def test_elastic_constants_solver():
    """
    @brief 测试弹性常数求解器的功能。
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
    solver = ElasticConstantsSolver()
    C = solver.solve(strains, stresses)
    # 检查 C 是否为 6x6 矩阵
    assert C.shape == (6, 6), "Elastic constants matrix shape mismatch."
    # 预期弹性常数矩阵（示例值）
    expected_C = np.array(
        [
            [69.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 69.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 69.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 23.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 23.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 23.0],
        ]
    )
    # 检查弹性常数矩阵是否接近预期值
    np.testing.assert_array_almost_equal(C, expected_C, decimal=2)


def test_force_direction():
    """
    @brief 验证力的方向是否为负梯度方向。
    """
    # 创建一个简单的晶胞
    atoms = [
        Atom(id=0, mass=26.9815, position=np.array([0.0, 0.0, 0.0]), symbol="Al"),
        Atom(id=1, mass=26.9815, position=np.array([2.55, 2.55, 2.55]), symbol="Al"),
    ]
    lattice_vectors = np.eye(3) * 5.1  # 示例晶格向量
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 定义 Lennard-Jones 势
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma  # Å
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    # 计算初始能量和力
    initial_energy = lj_potential.calculate_energy(cell)
    initial_force = cell.get_forces()

    # 计算能量的数值梯度近似
    delta = 1e-5
    expected_force = np.zeros_like(initial_force)
    for i in range(cell.num_atoms):
        for dim in range(3):
            # 正向微小位移
            displaced = cell.copy()
            displaced.atoms[i].position[dim] += delta
            energy_displaced = lj_potential.calculate_energy(displaced)
            # 负梯度近似
            expected_force[i, dim] = -(energy_displaced - initial_energy) / delta

    # 检查力方向是否接近负梯度方向
    np.testing.assert_array_almost_equal(initial_force, expected_force, decimal=3)