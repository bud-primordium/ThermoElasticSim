# tests/test_md_simulation.py

"""
@file test_md_simulation.py
@brief 测试 md_simulation.py 模块中的 MDSimulator 类。
"""

import unittest
from src.python.structure import CrystalStructure, Particle
from src.python.potentials import LennardJonesPotential
from src.python.md_simulation import MDSimulator


class TestMDSimulator(unittest.TestCase):
    """
    @class TestMDSimulator
    @brief 测试 MDSimulator 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        particles = [
            Particle(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0]),
            Particle(
                id=2, symbol="Al", mass=26.9815, position=[1.8075, 1.8075, 1.8075]
            ),
            # 添加更多粒子
        ]
        lattice_vectors = [[3.405, 0.0, 0.0], [0.0, 3.405, 0.0], [0.0, 0.0, 3.405]]
        crystal = CrystalStructure(lattice_vectors=lattice_vectors, particles=particles)
        potential = LennardJonesPotential(
            parameters={"epsilon": 0.0103, "sigma": 3.405}, cutoff=5.0
        )
        self.md_simulator = MDSimulator(
            crystal_structure=crystal,
            potential=potential,
            temperature=300.0,  # K
            pressure=0.0,  # GPa
            timestep=1.0e-3,  # ps
            thermostat="Nosé-Hoover",
            barostat="NoBarostat",
        )

    def test_run_simulation_not_implemented(self) -> None:
        """
        @brief 测试 MDSimulator.run_simulation 方法是否抛出 NotImplementedError。
        """
        with self.assertRaises(NotImplementedError):
            self.md_simulator.run_simulation(steps=1000)

    def test_collect_stress_not_implemented(self) -> None:
        """
        @brief 测试 MDSimulator.collect_stress 方法是否抛出 NotImplementedError。
        """
        with self.assertRaises(NotImplementedError):
            _ = self.md_simulator.collect_stress()


if __name__ == "__main__":
    unittest.main()
