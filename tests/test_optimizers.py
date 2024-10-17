# tests/test_optimizers.py

import pytest
import numpy as np
from python.optimizers import GradientDescentOptimizer
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential


@pytest.fixture
def optimizer():
    """
    @fixture 定义梯度下降优化器
    """
    return GradientDescentOptimizer(max_steps=1000, tol=1e-6, step_size=1e-3)


@pytest.fixture
def optimization_cell(pbc_enabled=False):
    """
    @fixture 创建一个简单的晶胞，包含两个原子
    """
    if pbc_enabled:
        lattice_vectors = np.eye(3) * 6.0  # Å, 以确保最小镜像距离 >= 2^(1/6)*sigma
    else:
        lattice_vectors = np.eye(3) * 1e8  # Å, 以禁用 PBC

    mass = 2.797909e-7  # eV·fs²/Å²
    position1 = np.array([0.0, 0.0, 0.0])
    position2 = np.array([2.55, 0.0, 0.0])  # 初始距离为 σ = 2.55 Å
    atom1 = Atom(id=0, symbol="Al", mass=mass, position=position1)
    atom2 = Atom(id=1, symbol="Al", mass=mass, position=position2)
    cell = Cell(lattice_vectors, [atom1, atom2], pbc_enabled=pbc_enabled)
    return cell


@pytest.fixture
def lj_potential_optim():
    """
    @fixture 定义 Lennard-Jones 势
    """
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    return LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)


def test_gradient_descent_optimizer(optimizer, optimization_cell, lj_potential_optim):
    """
    @brief 测试梯度下降优化器
    """
    optimizer.optimize(optimization_cell, lj_potential_optim)

    # 获取优化后的原子位置
    optimized_position1 = optimization_cell.atoms[0].position
    optimized_position2 = optimization_cell.atoms[1].position

    # 计算优化后的距离
    optimized_distance = np.linalg.norm(optimized_position2 - optimized_position1)

    # 预期优化后的距离应接近 2^(1/6) * sigma ≈ 2.857 Å
    equilibrium_distance = 2 ** (1 / 6) * lj_potential_optim.sigma
    assert np.isclose(optimized_distance, equilibrium_distance, atol=1e-3)


def test_optimizer_convergence(optimizer, optimization_cell, lj_potential_optim):
    """
    @brief 测试优化器是否能收敛
    """
    optimizer.optimize(optimization_cell, lj_potential_optim)

    # 计算优化后的距离
    optimized_distance = np.linalg.norm(
        optimization_cell.atoms[1].position - optimization_cell.atoms[0].position
    )

    # 预期优化后的距离应接近 2^(1/6) * sigma ≈ 2.857 Å
    equilibrium_distance = 2 ** (1 / 6) * lj_potential_optim.sigma
    assert np.abs(optimized_distance - equilibrium_distance) < 1e-3
