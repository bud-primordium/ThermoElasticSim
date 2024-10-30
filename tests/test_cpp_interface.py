# tests/test_cpp_interface.py

import pytest
import numpy as np
from itertools import combinations


def test_calculate_energy(lj_interface, two_atom_cell, two_atom_neighbor_list):
    """
    测试 C++ 实现的 Lennard-Jones 势能计算函数
    """
    num_atoms = two_atom_cell.num_atoms
    positions = two_atom_cell.get_positions().flatten()
    box_lengths = two_atom_cell.get_box_lengths()

    # 构建邻居对列表
    neighbor_pairs = [
        (i, j)
        for i, j in combinations(range(num_atoms), 2)
        if j in two_atom_neighbor_list.get_neighbors(i)
    ]
    neighbor_list_flat = [index for pair in neighbor_pairs for index in pair]
    neighbor_pairs_array = np.array(neighbor_list_flat, dtype=np.int32)
    num_pairs = len(neighbor_pairs)

    energy = lj_interface.calculate_lj_energy(
        num_atoms=num_atoms,
        positions=positions,
        epsilon=0.0103,
        sigma=2.55,
        cutoff=2.5 * 2.55,
        box_lengths=box_lengths,
        neighbor_pairs=neighbor_pairs_array,
        num_pairs=num_pairs,
    )

    # 检查能量是否为浮点数
    assert isinstance(energy, float), "Energy is not a float."

    # 根据 r = sigma, 势能应为 0
    expected_energy = 0.0

    assert np.isclose(
        energy, expected_energy, atol=1e-6
    ), f"Energy {energy} is not close to expected {expected_energy}."


@pytest.mark.parametrize(
    "r, expected_energy",
    [
        (1.0, 4.0 * 0.0103 * ((2.55 / 1.0) ** 12 - (2.55 / 1.0) ** 6)),
        (2.55, 4.0 * 0.0103 * ((2.55 / 2.55) ** 12 - (2.55 / 2.55) ** 6)),
        (3.0, -0.0096782),
    ],
)
def test_calculate_energy_different_r(lj_interface, r, expected_energy):
    """
    测试 C++ 实现的 Lennard-Jones 势能计算函数在不同距离下的能量
    """
    num_atoms = 2
    positions = np.array([0.0, 0.0, 0.0, r, 0.0, 0.0], dtype=np.float64)
    box_lengths = np.array([10.0, 10.0, 10.0], dtype=np.float64)

    # 构建邻居对列表
    neighbor_pairs = [(0, 1)]
    neighbor_list_flat = [0, 1]
    neighbor_pairs_array = np.array(neighbor_list_flat, dtype=np.int32)
    num_pairs = len(neighbor_pairs)

    energy = lj_interface.calculate_lj_energy(
        num_atoms=num_atoms,
        positions=positions,
        epsilon=0.0103,
        sigma=2.55,
        cutoff=2.5 * 2.55,
        box_lengths=box_lengths,
        neighbor_pairs=neighbor_pairs_array,
        num_pairs=num_pairs,
    )

    # 计算预期能量
    # expected_energy 已作为参数传入

    np.testing.assert_almost_equal(
        energy,
        expected_energy,
        decimal=6,
        err_msg=f"Energy at r={r} is not close to expected {expected_energy}.",
    )


def test_calculate_forces(lj_interface, two_atom_cell, two_atom_neighbor_list):
    """
    测试 C++ 实现的 Lennard-Jones 势能计算力函数
    """
    num_atoms = two_atom_cell.num_atoms
    positions = two_atom_cell.get_positions().flatten()
    box_lengths = two_atom_cell.get_box_lengths()
    forces = np.zeros_like(positions, dtype=np.float64)

    # 构建邻居对列表
    neighbor_pairs = [
        (i, j)
        for i, j in combinations(range(num_atoms), 2)
        if j in two_atom_neighbor_list.get_neighbors(i)
    ]
    neighbor_list_flat = [index for pair in neighbor_pairs for index in pair]
    neighbor_pairs_array = np.array(neighbor_list_flat, dtype=np.int32)
    num_pairs = len(neighbor_pairs)

    lj_interface.calculate_lj_forces(
        num_atoms=num_atoms,
        positions=positions,
        forces=forces,
        epsilon=0.0103,
        sigma=2.55,
        cutoff=2.5 * 2.55,
        box_lengths=box_lengths,
        neighbor_pairs=neighbor_pairs_array,
        num_pairs=num_pairs,
    )

    # 重新形状为 (num_atoms, 3)
    forces = forces.reshape((num_atoms, 3))

    # 计算期望的力
    epsilon = 0.0103
    sigma = 2.55
    r = sigma  # 两个原子之间的距离

    # 在 r = sigma, F = 24 * epsilon / sigma
    expected_force_magnitude = 24.0 * epsilon / sigma  # 0.2472 eV/Å

    # 计算力的方向
    expected_force_atom0 = np.array([-expected_force_magnitude, 0.0, 0.0])
    expected_force_atom1 = -expected_force_atom0

    # 检查力是否接近理论值
    np.testing.assert_allclose(
        forces[0],
        expected_force_atom0,
        atol=1e-6,
        err_msg=f"Force on atom0 {forces[0]} is not close to expected {expected_force_atom0}.",
    )
    np.testing.assert_allclose(
        forces[1],
        expected_force_atom1,
        atol=1e-6,
        err_msg=f"Force on atom1 {forces[1]} is not close to expected {expected_force_atom1}.",
    )
