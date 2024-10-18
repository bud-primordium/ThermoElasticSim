# tests/test_cpp_interface.py

import pytest
import numpy as np
from python.interfaces.cpp_interface import CppInterface
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential


@pytest.fixture
def lj_interface():
    return CppInterface("lennard_jones")


@pytest.fixture
def stress_calculator_interface():
    return CppInterface("stress_calculator")


@pytest.fixture
def nose_hoover_interface():
    return CppInterface("nose_hoover")


@pytest.fixture
def nose_hoover_chain_interface():
    return CppInterface("nose_hoover_chain")


def test_calculate_energy(lj_interface):
    """
    测试 C++ 实现的 Lennard-Jones 势能计算函数
    """
    num_atoms = 2
    positions = np.array(
        [0.0, 0.0, 0.0, 2.55, 0.0, 0.0], dtype=np.float64
    )  # 两个原子，距离 r = σ = 2.55 Å
    box_lengths = np.array([5.1, 5.1, 5.1], dtype=np.float64)  # 模拟盒子长度

    energy = lj_interface.calculate_energy(
        num_atoms,
        positions,
        epsilon=0.0103,
        sigma=2.55,
        cutoff=2.5 * 2.55,
        box_lengths=box_lengths,
    )

    # 检查能量是否为浮点数
    assert isinstance(energy, float), "Energy is not a float."

    # 根据 r = sigma, 势能应为 0
    expected_energy = 0.0

    assert np.isclose(
        energy, expected_energy, atol=1e-6
    ), f"Energy {energy} is not close to expected {expected_energy}."


def test_calculate_forces(lj_interface):
    """
    测试 C++ 实现的 Lennard-Jones 势能计算力函数
    """
    num_atoms = 2
    positions = np.array([0.0, 0.0, 0.0, 2.55, 0.0, 0.0], dtype=np.float64)
    box_lengths = np.array([5.1, 5.1, 5.1], dtype=np.float64)
    forces = np.zeros_like(positions, dtype=np.float64)

    lj_interface.calculate_forces(
        num_atoms,
        positions,
        forces,
        epsilon=0.0103,
        sigma=2.55,
        cutoff=2.5 * 2.55,
        box_lengths=box_lengths,
    )

    # 重新形状为 (num_atoms, 3)
    forces = forces.reshape((num_atoms, 3))

    # 计算期望的力
    epsilon = 0.0103

    # 理论上，在 r = sigma，力 = 24 * epsilon
    expected_force_theory = 24.0 * epsilon  # 0.2472 eV/Å

    # 检查力是否接近理论值
    np.testing.assert_allclose(forces[0, 0], expected_force_theory, atol=1e-6)
    np.testing.assert_allclose(forces[1, 0], -expected_force_theory, atol=1e-6)
    # 其他方向的力应为零
    np.testing.assert_allclose(forces[:, 1:], 0.0, atol=1e-6)


def test_nose_hoover(nose_hoover_interface):
    """
    测试 C++ 实现的 Nose-Hoover 恒温器函数
    """
    dt = 1.0
    num_atoms = 2
    masses = np.array([1.0, 1.0], dtype=np.float64)
    velocities = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    forces = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    xi = 0.0
    Q = 10.0
    target_temperature = 300.0

    # C++ 函数 expects a pointer to xi, so create a mutable array
    xi_array = np.array([xi], dtype=np.float64)

    updated_xi = nose_hoover_interface.nose_hoover(
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        xi_array,  # 传递数组
        Q,
        target_temperature,
    )

    # Check that xi has been updated
    assert isinstance(updated_xi, float), "xi is not a float."
    # 根据实现，xi 应该有变化
    assert (
        updated_xi != xi
    ), f"xi was not updated. Original: {xi}, Updated: {updated_xi}"


def test_nose_hoover_chain(nose_hoover_chain_interface):
    """
    测试 C++ 实现的 Nose-Hoover 链恒温器函数
    """
    dt = 1.0
    num_atoms = 2
    masses = np.array([1.0, 1.0], dtype=np.float64)
    velocities = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    forces = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    xi_chain = np.zeros(2, dtype=np.float64)  # 假设链长度为 2
    Q = np.array([10.0, 10.0], dtype=np.float64)
    chain_length = 2
    target_temperature = 300.0

    nose_hoover_chain_interface.nose_hoover_chain(
        dt,
        num_atoms,
        masses,
        velocities,
        forces,
        xi_chain,
        Q,
        chain_length,
        target_temperature,
    )

    # Check that xi_chain has been updated
    assert isinstance(xi_chain, np.ndarray), "xi_chain is not a numpy array."
    assert xi_chain.shape == (chain_length,), "xi_chain shape mismatch."
    # 检查是否有变化
    assert not np.all(xi_chain == 0.0), "xi_chain was not updated."
