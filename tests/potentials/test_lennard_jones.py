#!/usr/bin/env python3
"""
测试 Lennard-Jones 势
"""

import numpy as np
import pytest

from thermoelasticsim.core.structure import Atom, Cell
from thermoelasticsim.potentials.lennard_jones import LennardJonesPotential
from thermoelasticsim.utils.utils import NeighborList

try:
    import thermoelasticsim._cpp_core as _cpp_core  # 新 pybind11 模块

    HAS_PYBIND = True
except Exception:
    _cpp_core = None
    HAS_PYBIND = False

# 定义测试用的LJ势参数
EPSILON = 0.0103  # eV
SIGMA = 3.4  # Angstrom
CUTOFF = 8.5  # Angstrom


@pytest.fixture
def lj_potential():
    """提供一个LennardJonesPotential实例"""
    return LennardJonesPotential(epsilon=EPSILON, sigma=SIGMA, cutoff=CUTOFF)


@pytest.fixture
def two_atom_cell():
    """提供一个包含两个原子的晶胞，用于测试"""
    atom1 = Atom(id=1, symbol="Ar", mass_amu=39.948, position=np.array([0, 0, 0]))
    atom2 = Atom(
        id=2, symbol="Ar", mass_amu=39.948, position=np.array([3.8, 0, 0])
    )  # 略大于sigma
    cell_vectors = np.diag([20, 20, 20])
    cell = Cell(cell_vectors, [atom1, atom2])
    return cell


def test_lj_potential_initialization(lj_potential):
    """测试LJ势是否正确初始化"""
    assert lj_potential.parameters["epsilon"] == EPSILON
    assert lj_potential.parameters["sigma"] == SIGMA
    assert lj_potential.cutoff == CUTOFF


def test_lj_energy_calculation(lj_potential, two_atom_cell):
    """
    测试双原子体系的LJ势能计算。
    手动计算一个简单情况并与C++代码的结果对比。
    """
    neighbor_list = NeighborList(cutoff=CUTOFF)
    neighbor_list.build(two_atom_cell)

    # 手动计算理论能量
    r = 3.8
    sr6 = (SIGMA / r) ** 6
    sr12 = sr6**2
    expected_energy = 4 * EPSILON * (sr12 - sr6)

    # 从C++代码获取能量
    calculated_energy = lj_potential.calculate_energy(two_atom_cell, neighbor_list)

    assert calculated_energy == pytest.approx(expected_energy, rel=1e-5)


def test_lj_force_calculation(lj_potential, two_atom_cell):
    """
    测试双原子体系的LJ力计算。
    手动计算一个简单情况并与C++代码的结果对比。
    """
    neighbor_list = NeighborList(cutoff=CUTOFF)
    neighbor_list.build(two_atom_cell)

    # 手动计算理论力大小
    r = 3.8
    r_inv = 1.0 / r
    sr6 = (SIGMA * r_inv) ** 6
    sr12 = sr6**2
    # F = -dU/dr = 24 * epsilon / r * (2 * sr12 - sr6)
    expected_force_magnitude = 24 * EPSILON * r_inv * (2 * sr12 - sr6)

    # 从C++代码获取力
    lj_potential.calculate_forces(two_atom_cell, neighbor_list)

    force_on_atom1 = two_atom_cell.atoms[0].force
    force_on_atom2 = two_atom_cell.atoms[1].force

    # 验证力的方向和大小
    # 力应该沿着原子间连线方向
    assert np.abs(force_on_atom1[0]) == pytest.approx(
        expected_force_magnitude, rel=1e-5
    )
    assert np.allclose(force_on_atom1[1:], [0, 0])

    # 验证牛顿第三定律：F1 = -F2
    assert np.allclose(force_on_atom1, -force_on_atom2)
