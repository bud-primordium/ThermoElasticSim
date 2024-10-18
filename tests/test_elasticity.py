# tests/test_elasticity.py

import pytest
import numpy as np
from src.python.structure import Atom, Cell
from src.python.potentials import LennardJonesPotential
from src.python.elasticity import ElasticConstantsSolver, ElasticConstantsCalculator


def test_elastic_constants_calculator():
    """
    @brief 测试 ElasticConstantsCalculator 计算弹性常数
    """
    # 创建一个简单的晶胞，例如立方晶格
    atoms = [
        Atom(id=0, mass=26.9815, position=np.array([0.0, 0.0, 0.0]), symbol="Al"),
        Atom(id=1, mass=26.9815, position=np.array([2.55, 2.55, 2.55]), symbol="Al"),
    ]
    lattice_vectors = np.eye(3) * 5.1  # 示例晶格向量
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 定义 Lennard-Jones 势
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 8.0  # Å, 示例截断半径
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    # 创建 ElasticConstantsCalculator 实例
    elastic_calculator = ElasticConstantsCalculator(
        cell=cell, potential=lj_potential, delta=1e-3, optimizer_type="BFGS"
    )

    # 计算弹性常数
    C = elastic_calculator.calculate_elastic_constants()

    # 将弹性常数矩阵转换为 GPa
    # 假设单位转换正确，此处示例可能需要根据实际单位调整
    C_in_GPa = C * 160.21766208

    # 预期弹性常数矩阵（根据你的测试数据调整）
    expected_C = np.array(
        [
            [69.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 69.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 69.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 23.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 23.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 23.0],
        ]
    )  # 示例值，需根据实际情况调整

    # 检查弹性常数矩阵是否接近预期值
    np.testing.assert_array_almost_equal(C_in_GPa, expected_C, decimal=1)
