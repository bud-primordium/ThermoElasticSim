# tests/test_elasticity.py

import pytest
import numpy as np
from python.structure import Cell, Atom
from python.potentials import LennardJonesPotential
from python.elasticity import ElasticConstantsCalculator


def test_elastic_constants_calculator():
    """
    @brief 测试 ElasticConstantsCalculator 计算弹性常数
    """
    # 创建一个简单的晶胞，例如立方晶格
    atoms = [
        Atom(id=0, mass=2816.78346, position=np.array([0.0, 0.0, 0.0]), symbol="Al"),
        Atom(
            id=1, mass=2816.78346, position=np.array([2.55, 2.55, 2.55]), symbol="Al"
        ),  # sigma=2.55 Å
    ]
    lattice_vectors = np.eye(3) * 5.1  # 示例晶格向量
    cell = Cell(atoms=atoms, lattice_vectors=lattice_vectors)

    # 定义 Lennard-Jones 势
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 8.0  # Å, 示例截断半径
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    # 创建 ElasticConstantsCalculator 实例
    elastic_calculator = ElasticConstantsCalculator(
        cell=cell, potential=lj_potential, delta=1e-3
    )

    # 计算弹性常数
    C = elastic_calculator.calculate_elastic_constants()

    # 预期弹性常数矩阵（根据文献或已知值）
    # 请根据实际材料（例如铝）的弹性常数进行调整，以下为示例值
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
    assert np.allclose(
        C, expected_C, atol=1.0
    ), f"弹性常数矩阵不接近预期值。\n计算结果:\n{C}\n预期值:\n{expected_C}"
