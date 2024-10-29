# tests/test_potentials.py

import pytest
import numpy as np
from python.structure import Atom, Cell
from python.utils import NeighborList
from itertools import product


def test_force_calculation(lj_potential, two_atom_cell):
    """
    测试 Lennard-Jones 势的力计算功能
    """
    lj_potential.calculate_forces(two_atom_cell)
    # 检查力是否非零且相反
    force1 = two_atom_cell.atoms[0].force
    force2 = two_atom_cell.atoms[1].force
    np.testing.assert_array_almost_equal(force1, -force2, decimal=10)
    assert not np.allclose(force1, 0)

    # 检查系统总力为零
    total_force = force1 + force2
    np.testing.assert_allclose(total_force, np.zeros(3), atol=1e-10)


def test_energy_calculation(lj_potential, two_atom_cell):
    """
    测试 Lennard-Jones 势的能量计算功能
    """
    energy = lj_potential.calculate_energy(two_atom_cell)
    # 由于只有两个原子且对称位置，能量应等于 4 * epsilon * ( (sigma/r)^12 - (sigma/r)^6 )
    r = np.linalg.norm(
        two_atom_cell.atoms[1].position - two_atom_cell.atoms[0].position
    )
    expected_energy = (
        4.0
        * lj_potential.parameters["epsilon"]
        * (
            (lj_potential.parameters["sigma"] / r) ** 12
            - (lj_potential.parameters["sigma"] / r) ** 6
        )
    )
    np.testing.assert_almost_equal(energy, expected_energy, decimal=10)


def test_neighbor_list_generation():
    """
    测试 NeighborList 的生成是否正确。
    """
    lattice_vectors = np.eye(3) * 10.0
    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[1.0, 1.0, 1.0]),
        Atom(id=1, symbol="Al", mass_amu=26.9815, position=[3.0, 1.0, 1.0]),
        Atom(id=2, symbol="Al", mass_amu=26.9815, position=[1.0, 3.0, 1.0]),
        Atom(id=3, symbol="Al", mass_amu=26.9815, position=[1.0, 1.0, 3.0]),
    ]
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    cutoff = 2.5 * 2.55
    neighbor_list = NeighborList(cutoff=cutoff)
    neighbor_list.build(cell)

    # 原子 0 应该与 1, 2, 3 相邻
    expected_neighbors_0 = [1, 2, 3]
    assert (
        sorted(neighbor_list.get_neighbors(0)) == expected_neighbors_0
    ), "Atom 0 neighbors incorrect."


@pytest.mark.parametrize(
    "r, expected_energy",
    [
        (r, 4.0 * 0.0103 * ((2.55 / r) ** 12 - (2.55 / r) ** 6))
        for r in np.arange(2.35, 2.75, 0.1)
    ],
)
def test_lj_potential_with_neighbor_list(lj_potential, r, expected_energy):
    """
    测试 Lennard-Jones 势是否正确使用邻居列表计算力和能量。
    """
    # 创建两个原子，距离为 r
    lattice_vectors = np.eye(3) * 10.0
    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[1.0, 1.0, 1.0]),
        Atom(id=1, symbol="Al", mass_amu=26.9815, position=[1.0 + r, 1.0, 1.0]),
    ]
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    neighbor_list = NeighborList(cutoff=lj_potential.cutoff)
    neighbor_list.build(cell)
    lj_potential.set_neighbor_list(neighbor_list)

    # 计算力
    lj_potential.calculate_forces(cell)

    # 检查力是否正确（由于对称性和力的牛顿第三定律）
    forces = cell.get_forces()
    assert forces.shape == (2, 3), "Force array shape incorrect."

    # 计算预期力
    epsilon = 0.0103
    sigma = 2.55
    expected_force_magnitude = (
        24.0 * epsilon * ((2 * (sigma / r) ** 12) - (sigma / r) ** 6) / r
    )
    expected_force_atom0 = np.array([-expected_force_magnitude, 0.0, 0.0])
    expected_force_atom1 = -expected_force_atom0

    # 检查力是否接近理论值
    np.testing.assert_allclose(
        forces[0],
        expected_force_atom0,
        atol=1e-2,
        err_msg=f"Force on atom0 at r={r} is not close to expected {expected_force_atom0}.",
    )
    np.testing.assert_allclose(
        forces[1],
        expected_force_atom1,
        atol=1e-2,
        err_msg=f"Force on atom1 at r={r} is not close to expected {expected_force_atom1}.",
    )

    # 计算能量
    energy = lj_potential.calculate_energy(cell)
    np.testing.assert_almost_equal(
        energy,
        expected_energy,
        decimal=6,
        err_msg=f"Energy at r={r} is not close to expected {expected_energy}.",
    )
