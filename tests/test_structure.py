# tests/test_structure.py

"""
@file test_structure.py
@brief 测试 structure.py 模块中的 Particle 和 CrystalStructure 类。
"""

import unittest
import numpy as np
from src.python.structure import Particle, CrystalStructure


class TestParticle(unittest.TestCase):
    """
    @class TestParticle
    @brief 测试 Particle 类。
    """

    def test_particle_initialization(self) -> None:
        """
        @brief 测试粒子初始化是否正确。
        """
        p = Particle(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0])
        self.assertEqual(p.id, 1)
        self.assertEqual(p.symbol, "Al")
        self.assertAlmostEqual(p.mass, 26.9815)
        np.testing.assert_array_equal(p.position, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(p.velocity, np.zeros(3))

    def test_update_position(self) -> None:
        """
        @brief 测试更新粒子位置。
        """
        p = Particle(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0])
        p.update_position([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(p.position, np.array([1.0, 1.0, 1.0]))

    def test_update_velocity(self) -> None:
        """
        @brief 测试更新粒子速度。
        """
        p = Particle(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0])
        p.update_velocity([0.5, 0.5, 0.5])
        np.testing.assert_array_equal(p.velocity, np.array([0.5, 0.5, 0.5]))


class TestCrystalStructure(unittest.TestCase):
    """
    @class TestCrystalStructure
    @brief 测试 CrystalStructure 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        self.particles = [
            Particle(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0]),
            Particle(
                id=2, symbol="Al", mass=26.9815, position=[1.8075, 1.8075, 1.8075]
            ),
        ]
        self.lattice_vectors = [[3.615, 0.0, 0.0], [0.0, 3.615, 0.0], [0.0, 0.0, 3.615]]
        self.crystal = CrystalStructure(
            lattice_vectors=self.lattice_vectors, particles=self.particles
        )

    def test_volume_calculation(self) -> None:
        """
        @brief 测试晶胞体积计算。
        """
        expected_volume = 3.615**3
        self.assertAlmostEqual(self.crystal.volume, expected_volume)

    def test_apply_deformation(self) -> None:
        """
        @brief 测试施加应变后的晶体结构更新。
        """
        F = np.array([[1.01, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.crystal.apply_deformation(F)
        expected_lattice_vectors = np.dot(F, np.array(self.lattice_vectors))
        np.testing.assert_array_almost_equal(
            self.crystal.lattice_vectors, expected_lattice_vectors
        )
        for i, particle in enumerate(self.crystal.particles):
            expected_position = np.dot(F, np.array(self.particles[i].position))
            np.testing.assert_array_almost_equal(particle.position, expected_position)


if __name__ == "__main__":
    unittest.main()
