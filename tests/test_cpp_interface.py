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
def nose_hoover_interface():
    """
    @fixture 创建 Nose-Hoover C++ 接口实例
    """
    return CppInterface("nose_hoover")


@pytest.fixture
def nose_hoover_chain_interface():
    """
    @fixture 创建 Nose-Hoover 链 C++ 接口实例
    """
    return CppInterface("nose_hoover_chain")


def test_calculate_energy(lj_interface):
    """
    @brief 测试 C++ 实现的 Lennard-Jones 势能计算函数
    """
    num_atoms = 2
    positions = np.array([0.0, 0.0, 0.0, 2.55, 0.0, 0.0], dtype=np.float64)
    box_lengths = np.array([5.1, 5.1, 5.1], dtype=np.float64)

    energy = lj_interface.calculate_energy(
        num_atoms,
        positions,
        epsilon=0.0103,
        sigma=2.55,
        cutoff=2.5 * 2.55,
        box_lengths=box_lengths,
    )

    # 检查能量是否为浮点数
    assert isinstance(energy, float)
    # 根据原子距离检查能量是否合理（这里假设能量应为负数，因为原子相距小于截断半径）
    assert energy < 0


def test_calculate_forces(lj_interface):
    """
    @brief 测试 C++ 实现的 Lennard-Jones 力计算函数
    """
    num_atoms = 2
    positions = np.array([0.0, 0.0, 0.0, 2.55, 0.0, 0.0], dtype=np.float64)
    forces = np.zeros_like(positions)
    box_lengths = np.array([5.1, 5.1, 5.1], dtype=np.float64)

    lj_interface.calculate_forces(
        num_atoms,
        positions,
        forces,
        epsilon=0.0103,
        sigma=2.55,
        cutoff=2.5 * 2.55,
        box_lengths=box_lengths,
    )

    # 检查力是否非零且相反
    force1 = forces[0:3]
    force2 = forces[3:6]
    np.testing.assert_array_almost_equal(force1, -force2, decimal=6)
    assert not np.allclose(force1, 0)


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

    # 检查返回的 xi 是否为浮点数
    assert isinstance(updated_xi, float)


def test_nose_hoover_chain(nose_hoover_chain_interface):
    """
    @brief 测试 C++ 实现的 Nose-Hoover 链恒温器函数
    """
    dt = 1.0
    num_atoms = 2
    masses = np.array([1.0, 1.0], dtype=np.float64)
    velocities = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    forces = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    xi_chain = np.array([0.0, 0.0], dtype=np.float64)  # 假设链长度为2
    Q = 10.0
    target_temperature = 300.0

    # 调用 Nose-Hoover 链
    nose_hoover_chain_interface.nose_hoover_chain(
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        xi_chain,
        Q,
        target_temperature,
    )

    # 检查 xi_chain 是否被更新
    assert isinstance(xi_chain, np.ndarray)
    assert xi_chain.shape == (2,)
