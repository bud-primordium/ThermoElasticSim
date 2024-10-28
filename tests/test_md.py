# tests/test_md.py

import pytest
import numpy as np

from python.md_simulator import MDSimulator
from python.integrators import VelocityVerletIntegrator
from python.thermostats import NoseHooverThermostat

from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.utils import NeighborList


@pytest.fixture
def simple_cell(pbc_enabled=True):
    """
    @fixture 创建一个简单的晶胞，包含两个原子。
    """
    lattice_vectors = np.eye(3) * 6.0 if pbc_enabled else np.eye(3) * 1e8  # Å
    mass_amu = 26.9815  # amu (Aluminum)
    position1 = np.array([0.0, 0.0, 0.0])
    position2 = np.array([2.55, 0.0, 0.0])  # 与原子1相距 σ = 2.55 Å
    atom1 = Atom(id=0, symbol="Al", mass_amu=mass_amu, position=position1)
    atom2 = Atom(id=1, symbol="Al", mass_amu=mass_amu, position=position2)
    cell = Cell(
        lattice_vectors=lattice_vectors, atoms=[atom1, atom2], pbc_enabled=pbc_enabled
    )
    return cell


@pytest.fixture
def lj_potential():
    """
    @fixture 定义 Lennard-Jones 势。
    """
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    return LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)


@pytest.fixture
def integrator():
    """
    @fixture 定义积分器。
    """
    return VelocityVerletIntegrator()


@pytest.fixture
def thermostat():
    """
    @fixture 定义恒温器。
    """
    return NoseHooverThermostat(target_temperature=300, time_constant=100)


@pytest.fixture
def lj_potential_with_neighbors(lj_potential, simple_cell):
    """
    @fixture 定义 Lennard-Jones 势，并设置邻居列表。
    """
    neighbor_list = NeighborList(cutoff=lj_potential.cutoff)
    neighbor_list.build(simple_cell)
    lj_potential.set_neighbor_list(neighbor_list)
    return lj_potential


def test_md_simulation(simple_cell, lj_potential_with_neighbors, integrator):
    """
    @brief 测试分子动力学模拟器的运行。
    """
    md_simulator = MDSimulator(
        cell=simple_cell,
        potential=lj_potential_with_neighbors,
        integrator=integrator,
        thermostat=None,
    )
    md_simulator.run(steps=10, dt=1.0)  # dt 单位为 fs

    # 检查原子的位置和速度是否发生变化
    atom1 = simple_cell.atoms[0]
    atom2 = simple_cell.atoms[1]
    assert not np.allclose(atom1.position, [0.0, 0.0, 0.0]), "原子1的位置未发生变化。"
    assert not np.allclose(atom2.position, [2.55, 0.0, 0.0]), "原子2的位置未发生变化。"
    assert not np.allclose(atom1.velocity, [0.0, 0.0, 0.0]), "原子1的速度未发生变化。"
    assert not np.allclose(atom2.velocity, [0.0, 0.0, 0.0]), "原子2的速度未发生变化。"


def test_md_simulation_with_thermostat(
    simple_cell, lj_potential_with_neighbors, integrator, thermostat
):
    """
    @brief 测试分子动力学模拟器的运行，带恒温器。
    """
    md_simulator = MDSimulator(
        cell=simple_cell,
        potential=lj_potential_with_neighbors,
        integrator=integrator,
        thermostat=thermostat,
    )
    md_simulator.run(steps=10, dt=1.0)  # dt 单位为 fs

    # 检查原子的位置和速度是否发生变化
    atom1 = simple_cell.atoms[0]
    atom2 = simple_cell.atoms[1]
    assert not np.allclose(atom1.position, [0.0, 0.0, 0.0]), "原子1的位置未发生变化。"
    assert not np.allclose(atom2.position, [2.55, 0.0, 0.0]), "原子2的位置未发生变化。"
    assert not np.allclose(atom1.velocity, [0.0, 0.0, 0.0]), "原子1的速度未发生变化。"
    assert not np.allclose(atom2.velocity, [0.0, 0.0, 0.0]), "原子2的速度未发生变化。"

    # 检查 xi 是否被更新
    assert thermostat.xi[0] != 0.0, "xi 未被更新。"
