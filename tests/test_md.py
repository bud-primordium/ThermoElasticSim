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

from copy import deepcopy  # 用于深拷贝对象


# 定义 integrator fixture
@pytest.fixture
def integrator():
    """
    创建一个 Velocity Verlet 积分器实例。
    """
    return VelocityVerletIntegrator(dt=1.0)  # dt 单位为 fs


# 定义 thermostat fixture
@pytest.fixture
def thermostat():
    """
    创建一个 Nose-Hoover 恒温器实例。
    """
    return NoseHooverThermostat(temperature=300.0, Q=100.0)


# 定义 Lennard-Jones 势能对象的 fixture，并设置邻居列表
@pytest.fixture
def lj_potential_with_neighbor_list(simple_cell, lj_potential):
    """
    创建一个 Lennard-Jones 势能对象，并设置邻居列表，用于简单晶胞的分子动力学测试。
    """
    # 深拷贝 Lennard-Jones 势能对象，以避免影响其他测试
    lj_potential_copy = deepcopy(lj_potential)

    # 创建并构建邻居列表
    neighbor_list = NeighborList(cutoff=lj_potential_copy.cutoff)
    neighbor_list.build(simple_cell)
    lj_potential_copy.set_neighbor_list(neighbor_list)
    return lj_potential_copy


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
