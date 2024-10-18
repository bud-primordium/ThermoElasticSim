# tests/test_optimizers.py

import pytest
import numpy as np
import logging
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.optimizers import GradientDescentOptimizer, BFGSOptimizer
from python.elasticity import ElasticConstantsSolver  # 确保导入
from python.utils import AMU_TO_EVFSA2


@pytest.fixture
def lj_potential_optim():
    return LennardJonesPotential(epsilon=0.0103, sigma=2.55, cutoff=2.5 * 2.55)


def test_gradient_descent_optimizer(lj_potential_optim):
    """
    @brief 测试梯度下降优化器。
    """
    logger = logging.getLogger(__name__)
    optimizer = GradientDescentOptimizer(max_steps=100, tol=1e-5, step_size=1e-3)
    # 创建一个简单的晶胞
    atoms = [
        Atom(
            id=0,
            mass=26.9815 * AMU_TO_EVFSA2,
            position=np.array([0.0, 0.0, 0.0]),
            symbol="Al",
        ),
        Atom(
            id=1,
            mass=26.9815 * AMU_TO_EVFSA2,
            position=np.array([2.55, 0.0, 0.0]),
            symbol="Al",
        ),
    ]
    lattice_vectors = np.eye(3) * 5.1
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    optimizer.optimize(cell, lj_potential_optim)
    # 检查优化是否收敛
    assert optimizer.converged, "梯度下降优化器未收敛"

    # 输出优化后的能量和力
    energy = lj_potential_optim.calculate_energy(cell)
    forces = cell.get_forces()
    logger.debug(f"Post-optimization Energy: {energy:.6f} eV")
    logger.debug(f"Post-optimization Forces: {forces}")


def test_bfgs_optimizer(lj_potential_optim):
    """
    @brief 测试 BFGS 优化器。
    """
    logger = logging.getLogger(__name__)
    optimizer = BFGSOptimizer(tol=1e-5)
    # 创建一个简单的晶胞
    atoms = [
        Atom(
            id=0,
            mass=26.9815 * AMU_TO_EVFSA2,
            position=np.array([0.0, 0.0, 0.0]),
            symbol="Al",
        ),
        Atom(
            id=1,
            mass=26.9815 * AMU_TO_EVFSA2,
            position=np.array([2.55, 0.0, 0.0]),
            symbol="Al",
        ),
    ]
    lattice_vectors = np.eye(3) * 5.1
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    optimizer.optimize(cell, lj_potential_optim)
    # 检查优化是否收敛
    assert optimizer.converged, "BFGS 优化器未收敛"

    # 输出优化后的能量和力
    energy = lj_potential_optim.calculate_energy(cell)
    forces = cell.get_forces()
    logger.debug(f"Post-optimization Energy: {energy:.6f} eV")
    logger.debug(f"Post-optimization Forces: {forces}")


def test_optimizer_convergence():
    """
    @brief 测试优化器的收敛性。
    """
    optimizer = GradientDescentOptimizer(max_steps=1000, tol=1e-6, step_size=1e-3)
    # 创建一个简单的晶胞
    atoms = [
        Atom(
            id=0,
            mass=26.9815 * AMU_TO_EVFSA2,
            position=np.array([0.0, 0.0, 0.0]),
            symbol="Al",
        ),
        Atom(
            id=1,
            mass=26.9815 * AMU_TO_EVFSA2,
            position=np.array([2.55, 0.0, 0.0]),
            symbol="Al",
        ),
    ]
    lattice_vectors = np.eye(3) * 5.1
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    lj_potential = LennardJonesPotential(epsilon=0.0103, sigma=2.55, cutoff=2.5 * 2.55)
    optimizer.optimize(cell, lj_potential)
    # 检查优化是否收敛
    assert optimizer.converged, "优化器未能收敛"
