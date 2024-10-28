# 文件名: tests/test_md.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 测试分子动力学模拟器的功能，包括带恒温器和不带恒温器的情况。

import pytest
import numpy as np

from python.md_simulator import MDSimulator
from python.integrators import VelocityVerletIntegrator
from python.thermostats import NoseHooverThermostat

from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.utils import NeighborList

from conftest import generate_fcc_positions  # 从 conftest 导入


# 定义缺失的 integrator fixture
@pytest.fixture
def integrator():
    """
    创建一个 Velocity Verlet 积分器实例。
    """
    return VelocityVerletIntegrator(dt=1.0)  # dt 单位为 fs


# 定义缺失的 thermostat fixture
@pytest.fixture
def thermostat():
    """
    创建一个 Nose-Hoover 恒温器实例。
    """
    return NoseHooverThermostat(temperature=300.0, Q=100.0)


# 定义 simple_cell fixture，用于包含两个原子的简单晶胞
@pytest.fixture
def simple_cell():
    """
    创建一个简单的包含两个原子的晶胞，用于测试分子动力学模拟。
    """
    atoms = [
        Atom(
            id=0,
            symbol="Al",
            mass_amu=26.9815,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
        ),
        Atom(
            id=1,
            symbol="Al",
            mass_amu=26.9815,
            position=np.array([2.55, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
        ),
    ]
    lattice_vectors = np.eye(3) * 5.0  # Arbitrary lattice vectors
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    return cell


# 定义 Lennard-Jones 势能对象的 fixture
@pytest.fixture
def lj_potential_with_neighbor_list():
    """
    创建一个 Lennard-Jones 势能对象，并设置邻居列表。
    """
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma  # 截断半径
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    # 创建并构建邻居列表
    neighbor_list = NeighborList(cutoff=lj_potential.cutoff)
    neighbor_list.build(simple_cell)
    lj_potential.set_neighbor_list(neighbor_list)
    return lj_potential


def test_md_simulation(simple_cell, lj_potential_with_neighbor_list, integrator):
    """
    测试分子动力学模拟器的运行。
    """
    md_simulator = MDSimulator(
        cell=simple_cell,
        potential=lj_potential_with_neighbor_list,
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
    simple_cell, lj_potential_with_neighbor_list, integrator, thermostat
):
    """
    测试分子动力学模拟器的运行，带恒温器。
    """
    md_simulator = MDSimulator(
        cell=simple_cell,
        potential=lj_potential_with_neighbor_list,
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
