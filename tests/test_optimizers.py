import pytest
import numpy as np
import logging
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.optimizers import (
    GradientDescentOptimizer,
    BFGSOptimizer,
    LBFGSOptimizer,
)


# 配置日志
@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """
    配置日志以在测试期间输出到控制台和文件。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 控制台日志级别
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 创建文件处理器
    fh = logging.FileHandler("test_optimizers.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件日志级别
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    yield


@pytest.fixture
def lj_potential_optim():
    """
    创建一个 Lennard-Jones 势能对象，用于优化测试。
    """
    return LennardJonesPotential(epsilon=0.0103, sigma=2.55, cutoff=2.5 * 2.55)


@pytest.fixture
def cell_with_multiple_atoms():
    """
    创建一个包含多个原子的晶胞，模拟更复杂的系统。
    """
    sigma = 2.55
    r_m = 2 ** (1 / 6) * sigma
    delta = 0.1  # Å

    atoms = [
        Atom(
            id=i,
            mass_amu=26.9815,
            position=np.array([i * (r_m + delta), 0.0, 0.0]),
            symbol="Al",
        )
        for i in range(4)  # 增加多个原子
    ]
    lattice_vectors = np.eye(3) * (sigma * 6)  # 增加晶胞大小
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=False)
    return cell


def test_gradient_descent_optimizer(lj_potential_optim, cell_with_multiple_atoms):
    """
    测试梯度下降优化器，使用多个原子，无周期性边界条件。
    """
    logger = logging.getLogger(__name__)
    optimizer = GradientDescentOptimizer(
        max_steps=10000, tol=1e-3, step_size=1e-2, energy_tol=1e-4
    )
    cell = cell_with_multiple_atoms

    optimizer.optimize(cell, lj_potential_optim)

    assert optimizer.converged, "Gradient Descent Optimizer did not converge"

    energy = lj_potential_optim.calculate_energy(cell)
    forces = cell.get_forces()
    logger.debug(f"Gradient Descent - Post-optimization Energy: {energy:.6f} eV")
    logger.debug(f"Gradient Descent - Post-optimization Forces: {forces}")


# def test_bfgs_optimizer(lj_potential_optim, cell_with_multiple_atoms):
#     """
#     @brief 测试 BFGS 优化器。
#     """
#     logger = logging.getLogger(__name__)
#     optimizer = BFGSOptimizer(tol=1e-3, maxiter=10000)
#     cell = cell_with_multiple_atoms

#     logger.debug("开始 BFGS 优化测试...")

#     optimizer.optimize(cell, lj_potential_optim)

#     # 检查优化是否收敛
#     assert optimizer.converged, "BFGS Optimizer did not converge"

#     # 输出优化后的能量和力
#     energy = lj_potential_optim.calculate_energy(cell)
#     forces = cell.get_forces()
#     logger.debug(f"BFGS Optimizer - Post-optimization Energy: {energy:.6f} eV")
#     logger.debug(f"BFGS Optimizer - Post-optimization Forces: {forces}")

#     logger.debug("BFGS 优化测试结束。")


# def test_lbfgs_optimizer(lj_potential_optim, cell_with_multiple_atoms):
#     """
#     @brief 测试 L-BFGS 优化器。
#     """
#     logger = logging.getLogger(__name__)
#     optimizer = LBFGSOptimizer(tol=1e-3, maxiter=10000)
#     cell = cell_with_multiple_atoms

#     logger.debug("开始 L-BFGS 优化测试...")

#     optimizer.optimize(cell, lj_potential_optim)

#     # 检查优化是否收敛
#     assert optimizer.converged, "L-BFGS Optimizer did not converge"

#     # 输出优化后的能量和力
#     energy = lj_potential_optim.calculate_energy(cell)
#     forces = cell.get_forces()
#     logger.debug(f"L-BFGS Optimizer - Post-optimization Energy: {energy:.6f} eV")
#     logger.debug(f"L-BFGS Optimizer - Post-optimization Forces: {forces}")

#     logger.debug("L-BFGS 优化测试结束。")
