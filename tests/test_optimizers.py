# 文件名: tests/test_optimizers.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 测试不同优化器（梯度下降和 BFGS）在面心立方 (FCC) 结构上的性能。

import pytest
import numpy as np
import logging
from python.potentials import LennardJonesPotential
from python.structure import Atom, Cell
from python.optimizers import (
    GradientDescentOptimizer,
    BFGSOptimizer,
)
from python.utils import NeighborList


# 配置日志（假设已在 conftest.py 中通过 autouse=True 自动应用，不需在此定义）


# 定义 Lennard-Jones 势能对象的 fixture，并设置邻居列表
@pytest.fixture
def lj_potential_with_neighbor_list_optim(two_atom_cell):
    """
    创建一个 Lennard-Jones 势能对象，并设置邻居列表，用于优化测试。
    """
    # 创建新的 Lennard-Jones 势能对象
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    lj_potential_copy = LennardJonesPotential(
        epsilon=epsilon, sigma=sigma, cutoff=cutoff
    )

    # 创建并构建邻居列表
    neighbor_list = NeighborList(cutoff=lj_potential_copy.cutoff)
    neighbor_list.build(two_atom_cell)
    lj_potential_copy.set_neighbor_list(neighbor_list)

    return lj_potential_copy


@pytest.fixture
def fcc_1x1_cell():
    lattice_constant = 4.3  # Å (使用合适的FCC晶格常数)
    lattice_vectors = np.eye(3) * lattice_constant  # 正交晶格
    mass_amu = 26.9815  # amu (Aluminum)

    # 定义分数坐标并转换为真实坐标
    fractional_positions = [
        [0.0, 0.0, 0.0],  # 角位置
        [0.5, 0.5, 0.0],  # FCC 面心位置
        [0.5, 0.0, 0.5],  # FCC 面心位置
        [0.0, 0.5, 0.5],  # FCC 面心位置
    ]

    # 将分数坐标转换为真实坐标
    positions = [
        np.dot(lattice_vectors.T, frac_pos) for frac_pos in fractional_positions
    ]

    # 构建原子列表
    atoms = [
        Atom(id=i, symbol="Al", mass_amu=mass_amu, position=pos)
        for i, pos in enumerate(positions)
    ]

    # 创建Cell对象
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    return cell


def test_gradient_descent_optimizer(
    lj_potential_with_neighbor_list_optim, fcc_1x1_cell
):
    """
    测试梯度下降优化器，使用简fcc 1*1晶胞。
    """
    logger = logging.getLogger(__name__)
    optimizer = GradientDescentOptimizer(
        max_steps=2000, tol=1e-4, step_size=1e-4, energy_tol=1e-5
    )
    cell = fcc_1x1_cell.copy()  # 使用深拷贝以避免修改原始晶胞

    optimizer.optimize(cell, lj_potential_with_neighbor_list_optim)

    assert optimizer.converged, "Gradient Descent Optimizer did not converge"

    energy = lj_potential_with_neighbor_list_optim.calculate_energy(cell)
    forces = cell.get_forces()
    logger.debug(f"Gradient Descent - Post-optimization Energy: {energy:.6f} eV")
    logger.debug(f"Gradient Descent - Post-optimization Forces: {forces}")

    # 检查最大力是否小于容差
    max_force = max(np.linalg.norm(f) for f in forces)
    assert (
        max_force < optimizer.tol
    ), f"Max force {max_force} exceeds tolerance {optimizer.tol}"


def test_bfgs_optimizer(lj_potential_with_neighbor_list_optim, fcc_1x1_cell):
    """
    测试 BFGS 优化器，使用简fcc 1*1晶胞。
    """
    logger = logging.getLogger(__name__)
    optimizer = BFGSOptimizer(tol=1e-4, maxiter=2000)
    cell = fcc_1x1_cell.copy()  # 使用深拷贝以避免修改原始晶胞

    optimizer.optimize(cell, lj_potential_with_neighbor_list_optim)

    assert optimizer.converged, "BFGS Optimizer did not converge"

    energy = lj_potential_with_neighbor_list_optim.calculate_energy(cell)
    forces = cell.get_forces()
    logger.debug(f"BFGS Optimizer - Post-optimization Energy: {energy:.6f} eV")
    logger.debug(f"BFGS Optimizer - Post-optimization Forces: {forces}")

    # 检查最大力是否小于容差
    max_force = max(np.linalg.norm(f) for f in forces)
    assert (
        max_force < 2 * optimizer.tol
    ), f"Max force {max_force} exceeds tolerance {2 * optimizer.tol}"
