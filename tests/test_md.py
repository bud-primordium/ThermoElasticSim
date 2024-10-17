# tests/test_md.py

import unittest
import numpy as np
import sys
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.md import MDSimulator, VelocityVerletIntegrator, NoseHooverThermostat


class TestMD(unittest.TestCase):
    def setUp(self):
        # 创建一个包含两个原子的系统
        self.lattice_vectors = np.eye(3) * 4.05e-10
        mass = 26.9815 / (6.02214076e23) * 1e-3
        position1 = np.array([0.0, 0.0, 0.0])
        position2 = np.array([2.55e-10, 0.0, 0.0])  # 初始距离为 sigma
        atom1 = Atom(id=0, symbol="Al", mass=mass, position=position1)
        atom2 = Atom(id=1, symbol="Al", mass=mass, position=position2)
        self.cell = Cell(
            self.lattice_vectors, [atom1, atom2], pbc_enabled=False
        )  # 禁用周期性边界条件
        # 定义势能
        epsilon = 6.774e-21  # J
        sigma = 2.55e-10  # m
        cutoff = 2.5 * sigma
        self.potential = LennardJonesPotential(
            epsilon=epsilon, sigma=sigma, cutoff=cutoff
        )
        # 定义积分器和恒温器
        self.integrator = VelocityVerletIntegrator()
        self.thermostat = NoseHooverThermostat(
            target_temperature=300, time_constant=100
        )

    def test_md_simulation(self):
        md_simulator = MDSimulator(
            self.cell, self.potential, self.integrator, self.thermostat
        )
        md_simulator.run(steps=10, dt=1e-15)
        # 检查原子的位置和速度是否发生变化
        atom1 = self.cell.atoms[0]
        atom2 = self.cell.atoms[1]
        self.assertFalse(np.allclose(atom1.position, [0.0, 0.0, 0.0]))
        self.assertFalse(np.allclose(atom2.position, [2.55e-10, 0.0, 0.0]))
        self.assertFalse(np.allclose(atom1.velocity, [0.0, 0.0, 0.0]))
        self.assertFalse(np.allclose(atom2.velocity, [0.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
