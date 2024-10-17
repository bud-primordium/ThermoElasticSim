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
    masses = np.array([26.9815], dtype=np.float64)
    volume = 4.05**3
    epsilon = 0.0103
    sigma = 2.55
    cutoff = 2.5 * sigma
    lattice_vectors = np.eye(3, dtype=np.float64).flatten()

    stress_tensor = stress_interface.compute_stress(
        num_atoms,
        positions,
        velocities,
        forces,
        masses,
        volume,
        epsilon,
        sigma,
        cutoff,
        lattice_vectors,
    )

    # 检查应力张量是否为 3x3 矩阵
    assert stress_tensor.shape == (3, 3)
    # 由于没有力作用，应力张量应为零
    np.testing.assert_array_almost_equal(stress_tensor, np.zeros((3, 3)), decimal=6)


def test_nose_hoover(nose_hoover_interface):
    """
    @brief 测试 C++ 实现的 Nose-Hoover 恒温器函数
    """
    dt = 1.0
    num_atoms = 2
    masses = np.array([1.0, 1.0], dtype=np.float64)
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
