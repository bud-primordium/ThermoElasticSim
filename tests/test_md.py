# tests/test_md.py

import unittest
import numpy as np

from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.md import MDSimulator, VelocityVerletIntegrator, NoseHooverThermostat


class TestMD(unittest.TestCase):
    """
    @class TestMD
    @brief 测试分子动力学模拟器的功能
    """

    def setUp(self):
        # 创建一个简单的晶胞，包含两个原子
        lattice_vectors = np.eye(3) * 4.05  # Å
        mass = 26.9815  # 原子量，amu
        position1 = np.array([0.0, 0.0, 0.0])
        position2 = np.array([2.025, 0.0, 0.0])  # 与原子 1 相距一半的晶格常数
        atom1 = Atom(id=0, symbol="Al", mass=mass, position=position1)
        atom2 = Atom(id=1, symbol="Al", mass=mass, position=position2)
        self.cell = Cell(lattice_vectors, [atom1, atom2], pbc_enabled=True)

        # 定义 Lennard-Jones 势
        epsilon = 0.0103  # eV
        sigma = 2.55  # Å
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
        """
        @brief 测试分子动力学模拟器的运行
        """
        md_simulator = MDSimulator(
            self.cell, self.potential, self.integrator, self.thermostat
        )
        md_simulator.run(steps=10, dt=0.001)

        # 检查原子的位置和速度是否发生变化
        atom1 = self.cell.atoms[0]
        atom2 = self.cell.atoms[1]
        self.assertFalse(np.allclose(atom1.position, [0.0, 0.0, 0.0]))
        self.assertFalse(np.allclose(atom2.position, [2.025, 0.0, 0.0]))
        self.assertFalse(np.allclose(atom1.velocity, [0.0, 0.0, 0.0]))
        self.assertFalse(np.allclose(atom2.velocity, [0.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
