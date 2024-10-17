# tests/test_potentials.py

import pytest
import numpy as np
from python.potentials import LennardJonesPotential
from python.structure import Atom, Cell


@pytest.fixture
def lj_potential():
    """
    @fixture 定义 Lennard-Jones 势
    """
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    return LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)


@pytest.fixture
def two_atom_cell():
    """
    @fixture 创建一个简单的系统，包含两个原子
    """
    lattice_vectors = np.eye(3) * 4.05  # Å
    mass = 26.9815  # amu
    position1 = np.array([0.0, 0.0, 0.0])
    position2 = np.array([2.025, 0.0, 0.0])  # 2.025 Å
    atom1 = Atom(id=0, symbol="Al", mass=mass, position=position1)
    atom2 = Atom(id=1, symbol="Al", mass=mass, position=position2)
    cell = Cell(lattice_vectors, [atom1, atom2], pbc_enabled=True)
    return cell


def test_force_calculation(lj_potential, two_atom_cell):
    """
    @brief 测试 Lennard-Jones 势的力计算功能
    """
    lj_potential.calculate_forces(two_atom_cell)
    # 检查力是否非零且相反
    force1 = two_atom_cell.atoms[0].force
    force2 = two_atom_cell.atoms[1].force
    np.testing.assert_array_almost_equal(force1, -force2, decimal=10)
    assert not np.allclose(force1, 0)
