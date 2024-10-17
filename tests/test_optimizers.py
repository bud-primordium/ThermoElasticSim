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
    return GradientDescentOptimizer(max_steps=5000, tol=1e-8, step_size=0.1)


@pytest.fixture
def optimization_cell(pbc_enabled=False):
    """
    @fixture 创建一个简单的晶胞，包含两个原子
    """
    if pbc_enabled:
        lattice_vectors = np.eye(3) * 6.0  # Å, 以确保最小镜像距离 >= 2^(1/6)*sigma
    else:
        lattice_vectors = np.eye(3) * 1e8  # Å, 以禁用 PBC

    mass = 2816.78346  # eV·fs²/Å²
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
    assert np.isclose(optimized_distance, equilibrium_distance, atol=2e-3)


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
    assert np.abs(optimized_distance - equilibrium_distance) < 2e-3


def test_force_direction():
    """
    @brief 验证力的方向是否为负梯度方向
    """
    # 创建一个简单的晶胞
    atoms = [
        Atom(id=0, mass=2816.78346, position=np.array([0.0, 0.0, 0.0]), symbol="Al"),
        Atom(id=1, mass=2816.78346, position=np.array([2.55, 2.55, 2.55]), symbol="Al"),
    ]
    lattice_vectors = np.eye(3) * 5.1  # 示例晶格向量
    cell = Cell(atoms=atoms, lattice_vectors=lattice_vectors)

    # 定义 Lennard-Jones 势
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 8.0  # Å
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    # 计算力
    lj_potential.calculate_forces(cell)

    force_on_atom0 = cell.atoms[0].force
    force_on_atom1 = cell.atoms[1].force

    # 计算梯度方向并包括力的截断偏移
    r_vec = atoms[1].position - atoms[0].position
    r = np.linalg.norm(r_vec)
    if r >= cutoff:
        # 截断距离外，力为零
        expected_force = np.zeros(3)
    else:
        sr = sigma / r
        sr6 = sr**6
        sr12 = sr6**2
        F_LJ = 24.0 * epsilon * (2.0 * sr12 - sr6) / r
        sr_cutoff = sigma / cutoff
        sr6_cutoff = sr_cutoff**6
        sr12_cutoff = sr6_cutoff**2
        F_LJ_cutoff = 24.0 * epsilon * (2.0 * sr12_cutoff - sr6_cutoff) / cutoff
        F_total = F_LJ - F_LJ_cutoff
        expected_force = F_total * (r_vec / r)

    # 断言计算的力是否接近预期力
    assert np.allclose(
        force_on_atom0, expected_force, atol=1e-5
    ), f"Atom 0 的力方向错误: {force_on_atom0} != {expected_force}"
    assert np.allclose(
        force_on_atom1, -expected_force, atol=1e-5
    ), f"Atom 1 的力方向错误: {force_on_atom1} != {-expected_force}"
