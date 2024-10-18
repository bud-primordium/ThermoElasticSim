# tests/test_cpp_interface.py

import pytest
import numpy as np
from python.interfaces.cpp_interface import CppInterface


@pytest.fixture
def lj_interface():
    """
    @fixture 创建 Lennard-Jones C++ 接口实例
    """
    return CppInterface("lennard_jones")


@pytest.fixture
def stress_interface():
    """
    @fixture 创建 Stress Calculator C++ 接口实例
    """
    return CppInterface("stress_calculator")


@pytest.fixture
def nose_hoover_interface():
    """
    @fixture 创建 Nose-Hoover C++ 接口实例
    """
    return CppInterface("nose_hoover")


@pytest.fixture
def parrinello_rahman_hoover_interface():
    """
    @fixture 创建 Parrinello-Rahman-Hoover C++ 接口实例
    """
    return CppInterface("parrinello_rahman_hoover")


def test_calculate_forces(lj_interface):
    """
    @brief 测试 C++ 实现的 Lennard-Jones 力计算函数
    """
    num_atoms = 2
    positions = np.array([0.0, 0.0, 0.0, 2.55, 0.0, 0.0], dtype=np.float64)
    forces = np.zeros_like(positions)
    epsilon = 0.0103
    sigma = 2.55
    cutoff = 2.5 * sigma
    lattice_vectors = np.eye(3, dtype=np.float64).flatten()

    lj_interface.calculate_forces(
        num_atoms,
        positions,
        forces,
        epsilon,
        sigma,
        cutoff,
        lattice_vectors,
    )

    # 检查力是否非零且相反
    force1 = forces[0:3]
    force2 = forces[3:6]
    np.testing.assert_array_almost_equal(force1, -force2, decimal=6)
    assert not np.allclose(force1, 0)


def test_compute_stress(stress_interface):
    """
    @brief 测试 C++ 实现的应力计算函数
    """
    num_atoms = 1
    positions = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    velocities = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    forces = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    masses = np.array([26.9815], dtype=np.float64)  # amu
    volume = 4.05**3
    box_lengths = np.eye(3, dtype=np.float64).flatten()

    stress_tensor = np.zeros(9, dtype=np.float64)
    stress_interface.compute_stress(
        num_atoms,
        positions,
        velocities,
        forces,
        masses,
        volume,
        box_lengths,
        stress_tensor,
    )

    # 检查应力张量是否为 3x3 矩阵
    assert stress_tensor.shape == (9,)
    stress_tensor = stress_tensor.reshape((3, 3))
    # 由于没有力作用，应力张量应为零
    np.testing.assert_array_almost_equal(stress_tensor, np.zeros((3, 3)), decimal=6)


def test_nose_hoover(nose_hoover_interface):
    """
    @brief 测试 C++ 实现的 Nose-Hoover 恒温器函数
    """
    dt = 1.0
    num_atoms = 2
    masses = np.array([26.9815, 26.9815], dtype=np.float64)  # amu
    velocities = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    forces = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    xi = 0.0
    Q = 10.0
    target_temperature = 300.0

    # 调用 Nose-Hoover
    updated_xi = nose_hoover_interface.nose_hoover(
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        xi,
        Q,
        target_temperature,
    )

    # 由于初始 xi=0，动能接近 0，xi 应该有变化
    assert isinstance(updated_xi, float)
    assert not np.isclose(updated_xi, xi)


def test_parrinello_rahman_hoover(parrinello_rahman_hoover_interface):
    """
    @brief 测试 C++ 实现的 Parrinello-Rahman-Hoover 恒压器函数
    """
    dt = 1.0
    num_atoms = 2
    masses = np.array([26.9815, 26.9815], dtype=np.float64)  # amu
    velocities = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    forces = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    lattice_vectors = np.eye(3, dtype=np.float64).flatten()
    xi = np.zeros(6, dtype=np.float64)
    Q = np.ones(6, dtype=np.float64) * 10.0
    target_pressure = 1.0  # 示例压力

    # 调用 PRH
    parrinello_rahman_hoover_interface.parrinello_rahman_hoover(
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        lattice_vectors,
        xi,
        Q,
        target_pressure,
    )

    # 检查 xi 是否有变化
    assert isinstance(xi, np.ndarray)
    assert xi.shape == (6,)
    assert not np.allclose(xi, 0.0)

    # 检查 lattice_vectors 是否被更新（简单检查是否被缩放）
    # 由于 C++ 实现是简化版，这里只检查是否有缩放
    for i in range(3):
        for j in range(3):
            if i == j:
                assert lattice_vectors[i * 3 + j] != 1.0  # 原始值为1.0
            else:
                assert lattice_vectors[i * 3 + j] == 0.0  # 应保持为0
