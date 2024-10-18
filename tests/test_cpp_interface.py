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


def test_calculate_energy(lj_interface):
    """
    @brief 测试 C++ 实现的 Lennard-Jones 势能计算函数
    """
    num_atoms = 2
    positions = np.array([0.0, 0.0, 0.0, 2.55, 0.0, 0.0], dtype=np.float64)
    epsilon = 0.0103
    sigma = 2.55
    cutoff = 2.5 * sigma
    box_lengths = np.array([5.1, 5.1, 5.1], dtype=np.float64)

    energy = lj_interface.calculate_energy(
        num_atoms,
        positions,
        epsilon,
        sigma,
        cutoff,
        box_lengths,
    )

    # 检查能量是否为浮点数
    assert isinstance(energy, float)
    # 能量应为正数（根据初始位置，能量应大于零）
    assert energy > 0


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
    box_lengths = np.array([5.1, 5.1, 5.1], dtype=np.float64)

    lj_interface.calculate_forces(
        num_atoms,
        positions,
        forces,
        epsilon,
        sigma,
        cutoff,
        box_lengths,
    )

    # 检查力是否非零且相反
    force1 = forces[0:3]
    force2 = forces[3:6]
    np.testing.assert_array_almost_equal(force1, -force2, decimal=6)
    assert not np.allclose(force1, 0)
