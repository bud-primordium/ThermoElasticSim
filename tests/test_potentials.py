# tests/test_potentials.py

import pytest
import numpy as np
from python.structure import Atom, Cell
from python.utils import NeighborList
from itertools import product
from python.potentials import EAMAl1Potential


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


@pytest.fixture
def eam_al1_potential():
    """创建EAM Al1势的fixture"""
    return EAMAl1Potential()


@pytest.fixture
def two_al_atoms_cell():
    """创建含两个Al原子的晶胞fixture"""
    lattice_vectors = np.eye(3) * 10.0  # 足够大的盒子避免周期性影响
    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0]),
        Atom(
            id=1, symbol="Al", mass_amu=26.9815, position=[2.8, 0.0, 0.0]
        ),  # 近平衡距离
    ]
    return Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)


def test_eam_al1_force_calculation(eam_al1_potential, two_al_atoms_cell):
    """
    测试EAM Al1势的力计算
    - 验证牛顿第三定律
    - 验证总力为零
    """
    eam_al1_potential.calculate_forces(two_al_atoms_cell)

    force1 = two_al_atoms_cell.atoms[0].force
    force2 = two_al_atoms_cell.atoms[1].force

    # 验证力是否相等且方向相反
    np.testing.assert_array_almost_equal(force1, -force2, decimal=10)

    # 验证力不为零(因为不是平衡位置)
    assert not np.allclose(force1, 0)

    # 验证总力为零
    total_force = force1 + force2
    np.testing.assert_allclose(total_force, np.zeros(3), atol=1e-10)


def test_eam_al1_energy_calculation(eam_al1_potential, two_al_atoms_cell):
    """
    测试EAM Al1势的能量计算
    - 验证能量值的合理性
    - 验证能量随距离的变化趋势
    """
    energy = eam_al1_potential.calculate_energy(two_al_atoms_cell)

    # 验证能量为负值(结合能)
    assert energy < 0

    # 验证能量在合理范围内 (典型的Al原子对结合能在eV量级)
    assert -10.0 < energy < 0.0


def test_eam_al1_energy_force_distance_relationship(eam_al1_potential):
    """
    测试EAM Al1势能量和力随原子间距的变化关系
    - 验证在平衡位置附近的能量和力的行为
    - 验证在远距离势能趋近于零
    """
    lattice_vectors = np.eye(3) * 15.0
    distances = np.linspace(2.0, 6.0, 10)  # 测试不同的原子间距
    energies = []
    forces = []

    for r in distances:
        atoms = [
            Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0]),
            Atom(id=1, symbol="Al", mass_amu=26.9815, position=[r, 0.0, 0.0]),
        ]
        cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

        # 计算能量和力
        energies.append(eam_al1_potential.calculate_energy(cell))
        eam_al1_potential.calculate_forces(cell)
        forces.append(cell.atoms[0].force[0])  # x方向的力

    energies = np.array(energies)
    forces = np.array(forces)

    # 验证能量曲线存在最小值
    min_energy_idx = np.argmin(energies)
    assert 0 < min_energy_idx < len(distances) - 1

    # 验证在平衡位置处力接近于零
    min_force_idx = np.argmin(np.abs(forces))
    assert np.abs(forces[min_force_idx]) < 0.1

    # 验证在远距离处能量趋近于零
    assert np.abs(energies[-1]) < 0.1


def test_eam_al1_periodic_boundary(eam_al1_potential):
    """
    测试EAM Al1势在周期性边界条件下的行为
    """
    # 创建一个小的周期性盒子
    lattice_vectors = np.eye(3) * 5.0
    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0]),
        Atom(id=1, symbol="Al", mass_amu=26.9815, position=[4.8, 0.0, 0.0]),  # 接近边界
    ]
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 计算能量和力
    energy_pbc = eam_al1_potential.calculate_energy(cell)
    eam_al1_potential.calculate_forces(cell)
    force_pbc = cell.atoms[0].force

    # 创建等效的非周期性构型(原子实际距离相同)
    atoms_equiv = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0]),
        Atom(
            id=1, symbol="Al", mass_amu=26.9815, position=[0.2, 0.0, 0.0]
        ),  # 5.0-4.8=0.2
    ]
    cell_equiv = Cell(
        lattice_vectors=lattice_vectors, atoms=atoms_equiv, pbc_enabled=True
    )

    # 计算等效构型的能量和力
    energy_equiv = eam_al1_potential.calculate_energy(cell_equiv)
    eam_al1_potential.calculate_forces(cell_equiv)
    force_equiv = cell_equiv.atoms[0].force

    # 验证两种情况得到相同的能量和力
    np.testing.assert_allclose(energy_pbc, energy_equiv, rtol=1e-10)
    np.testing.assert_allclose(force_pbc, -force_equiv, rtol=1e-10)


import logging

logger = logging.getLogger(__name__)


def test_eam_forces_newton_third_law(eam_al1_potential):
    """测试EAM势的力是否满足牛顿第三定律"""
    # 创建两个原子的简单系统
    lattice_vectors = np.eye(3) * 10.0
    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0]),
        Atom(
            id=1, symbol="Al", mass_amu=26.9815, position=[2.86, 0.0, 0.0]
        ),  # 平衡位置附近
    ]
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 计算力
    eam_al1_potential.calculate_forces(cell)
    force1 = cell.atoms[0].force
    force2 = cell.atoms[1].force

    print(f"Force on atom 1: {force1}")
    print(f"Force on atom 2: {force2}")

    # 验证力是否相等且方向相反
    np.testing.assert_allclose(force1, -force2, rtol=1e-10)

    # 验证力的数量级是否合理（平衡位置附近应该很小）
    assert np.all(np.abs(force1) < 0.1)


def test_eam_forces_symmetry(eam_al1_potential):
    """测试EAM势的力是否具有正确的对称性"""
    lattice_vectors = np.eye(3) * 10.0
    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0]),
        Atom(
            id=1, symbol="Al", mass_amu=26.9815, position=[2.86, 0.0, 0.0]
        ),  # 平衡位置附近
    ]
    cell_x = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0]),
        Atom(id=1, symbol="Al", mass_amu=26.9815, position=[0.0, 2.86, 0.0]),  # y方向
    ]
    cell_y = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 计算力
    eam_al1_potential.calculate_forces(cell_x)
    force_x = cell_x.atoms[0].force[0]  # x方向的力

    eam_al1_potential.calculate_forces(cell_y)
    force_y = cell_y.atoms[0].force[1]  # y方向的力

    # 由于对称性，x和y方向的力应该相等
    np.testing.assert_allclose(force_x, force_y, rtol=1e-10)


def test_eam_al1_crystal_properties(eam_al1_potential):
    """
    测试EAM Al1势对Al晶体性质的预测
    - 验证fcc晶格常数
    - 验证内聚能
    """
    # 创建一个 FCC Al 单胞
    a0 = 4.046  # Al的实验晶格常数(Å)
    lattice_vectors = np.eye(3) * a0  # 使用单胞而不是2x2x2超胞

    # FCC的基本位置（以a0为单位）- 一个单胞包含4个原子
    fcc_positions = np.array(
        [
            [0.0, 0.0, 0.0],  # 顶角原子
            [0.5, 0.5, 0.0],  # 面心原子
            [0.5, 0.0, 0.5],  # 面心原子
            [0.0, 0.5, 0.5],  # 面心原子
        ]
    )

    # 调试输出
    logger.debug("\nFCC unit positions:")
    logger.debug(fcc_positions)

    # 创建原子列表
    atoms = []
    for i, pos in enumerate(fcc_positions):
        real_pos = pos * a0
        logger.debug(f"Atom {i} position: {real_pos}")
        atoms.append(Atom(id=i, symbol="Al", mass_amu=26.9815, position=real_pos))

    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 打印相邻原子间距（用于调试）
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = np.linalg.norm(atoms[i].position - atoms[j].position)
            logger.debug(f"Distance between atoms {i} and {j}: {dist:.3f} Å")

    # 计算内聚能
    energy = eam_al1_potential.calculate_energy(cell)
    cohesive_energy = energy / len(atoms)

    # 验证内聚能在合理范围内 (实验值约为-3.36 eV/atom)
    assert -3.5 < cohesive_energy < -2.0

    # 计算力
    eam_al1_potential.calculate_forces(cell)

    # 验证理想晶格中的力应该很小
    forces = cell.get_forces()
    assert np.all(np.abs(forces) < 0.01)
