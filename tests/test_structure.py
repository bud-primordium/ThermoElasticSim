# tests/test_structure.py

import pytest
import numpy as np
from python.structure import Atom, Cell


@pytest.fixture
def atom():
    """
    @fixture 创建一个原子实例
    """
    position = np.array([0.0, 0.0, 0.0])
    mass_amu = 26.9815  # amu
    return Atom(id=0, symbol="Al", mass_amu=mass_amu, position=position)


@pytest.fixture
def cell(atom):
    """
    @fixture 创建一个晶胞实例，包含一个原子
    """
    lattice_vectors = np.eye(3) * 4.05  # Å
    return Cell(lattice_vectors, [atom], pbc_enabled=True)


def test_atom_creation(atom):
    """
    @brief 测试原子的创建
    """
    assert atom.id == 0
    assert atom.symbol == "Al"
    np.testing.assert_array_equal(atom.position, np.array([0.0, 0.0, 0.0]))
    assert atom.mass_amu == 26.9815


def test_cell_creation(cell, atom):
    """
    @brief 测试晶胞的创建
    """
    np.testing.assert_array_equal(cell.lattice_vectors, np.eye(3) * 4.05)
    assert len(cell.atoms) == 1
    assert cell.atoms[0] == atom


def test_volume_calculation(cell):
    """
    @brief 测试晶胞体积的计算
    """
    expected_volume = np.linalg.det(cell.lattice_vectors)
    calculated_volume = cell.calculate_volume()
    assert np.isclose(calculated_volume, expected_volume)


def test_apply_periodic_boundary(cell):
    """
    @brief 测试周期性边界条件的应用
    """
    lattice_constant = cell.lattice_vectors[0, 0]  # 假设晶格为立方
    position = np.array(
        [lattice_constant + 1.0, -1.0, lattice_constant + 1.0]
    )  # 超出晶胞范围的坐标
    new_position = cell.apply_periodic_boundary(position)
    # 检查新位置是否在 [0, lattice_constant) 范围内
    assert np.all(new_position >= 0)
    assert np.all(new_position < lattice_constant)
