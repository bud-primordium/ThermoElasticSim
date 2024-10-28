# tests/test_md.py

import pytest
import numpy as np

from python.md_simulator import MDSimulator
from python.integrators import VelocityVerletIntegrator
from python.thermostats import NoseHooverThermostat

from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential


def test_md_simulation(simple_cell, lj_potential_with_neighbor_list, integrator):
    """
    @brief 测试分子动力学模拟器的运行。
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
    @brief 测试分子动力学模拟器的运行，带恒温器。
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
