# tests/test_mechanics.py

import pytest
import numpy as np
from python.mechanics import StressCalculatorLJ, StrainCalculator
from python.elasticity import ElasticConstantsSolver
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.utils import TensorConverter


@pytest.fixture
def single_atom_cell():
    """
    @fixture 创建一个简单的晶胞，包含一个原子。
    """
    lattice_vectors = np.eye(3) * 4.05  # Å
    mass = 26.9815  # 原子量，amu
    position = np.array([0.0, 0.0, 0.0])
    atom = Atom(id=0, symbol="Al", mass=mass, position=position)
    cell = Cell(lattice_vectors, [atom], pbc_enabled=True)
    return cell


@pytest.fixture
def lj_potential_single():
    """
    @fixture 定义 Lennard-Jones 势。
    """
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    return LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)


def test_stress_calculation(single_atom_cell, lj_potential_single):
    """
    @brief 测试应力计算器的功能。
    """
    stress_calculator = StressCalculatorLJ()
    # 计算力
    lj_potential_single.calculate_forces(single_atom_cell)
    # 计算应力
    stress_tensor = stress_calculator.compute_stress(
        single_atom_cell, lj_potential_single
    )
    # 检查应力张量是否为 3x3 矩阵
    assert stress_tensor.shape == (3, 3)
    # 由于只有一个原子且无力作用，应力张量应为零
    np.testing.assert_array_almost_equal(stress_tensor, np.zeros((3, 3)), decimal=6)


def test_strain_calculation():
    """
    @brief 测试应变计算器的功能。
    """
    strain_calculator = StrainCalculator()
    F = np.array([[1.01, 0, 0], [0, 1, 0], [0, 0, 1]])
    strain_voigt = strain_calculator.compute_strain(F)
    # 检查应变向量是否为 6 元素
    assert strain_voigt.shape == (6,)
    # 检查应变计算是否正确
    strain_tensor = 0.5 * (np.dot(F.T, F) - np.identity(3))
    expected_strain_voigt = TensorConverter.to_voigt(strain_tensor)
    expected_strain_voigt[3:] *= 2  # 剪切分量乘以 2
    np.testing.assert_array_almost_equal(strain_voigt, expected_strain_voigt, decimal=6)


def test_elastic_constants_solver():
    """
    @brief 测试弹性常数求解器的功能。
    """
    strains = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01]),
    ]
    stresses = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([69.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 69.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 69.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 23.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 23.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 23.0]),
    ]
    solver = ElasticConstantsSolver()
    C = solver.solve(strains, stresses)
    # 检查 C 是否为 6x6 矩阵
    assert C.shape == (6, 6)
    # 预期弹性常数矩阵（示例值）
    expected_C = np.array(
        [
            [69.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 69.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 69.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 23.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 23.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 23.0],
        ]
    )
    # 检查弹性常数矩阵是否接近预期值
    np.testing.assert_array_almost_equal(C, expected_C, decimal=2)
