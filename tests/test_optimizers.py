# tests/test_optimizers.py

import pytest
import numpy as np
import logging
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.optimizers import (
    GradientDescentOptimizer,
    BFGSOptimizer,
)
from datetime import datetime


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

    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = (
        f"./logs/test_optimizers_{current_time}.log"  # 生成带时间戳的日志文件名
    )

    # 创建文件处理器
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件日志级别
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    yield

    # 测试结束后移除处理器
    logger.removeHandler(ch)
    logger.removeHandler(fh)


@pytest.fixture
def lj_potential_optim():
    """
    创建一个 Lennard-Jones 势能对象，用于优化测试。
    """
    return LennardJonesPotential(epsilon=0.0103, sigma=2.55, cutoff=2.5 * 2.55)


@pytest.fixture
def cell_with_multiple_atoms():
    """
    创建一个包含多个原子的晶胞，模拟更复杂的系统（8个原子）。
    """
    sigma = 2.55
    r_m = 2 ** (1 / 6) * sigma  # 约为2.04 Å
    delta = 0.1  # Å

    num_repetitions = 1  # 1x1x1 单位晶胞，共8个原子（假设每个单位晶胞8个原子）
    lattice_constant = 5.1  # Å
    atoms = []
    positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
    ]  # 简单立方晶胞的8个原子位置

    for pos in positions:
        atoms.append(
            Atom(
                id=len(atoms),
                symbol="Al",
                mass_amu=26.9815,
                position=np.array(pos) * lattice_constant,
                velocity=None,
            )
        )

    lattice_vectors = np.eye(3) * lattice_constant  # 单位晶胞大小
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=False)
    return cell


def test_gradient_descent_optimizer(lj_potential_optim, cell_with_multiple_atoms):
    """
    测试梯度下降优化器，使用4个原子，无周期性边界条件。
    """
    logger = logging.getLogger(__name__)
    optimizer = GradientDescentOptimizer(
        max_steps=20000, tol=1e-3, step_size=1e-3, energy_tol=1e-4
    )
    cell = cell_with_multiple_atoms

    optimizer.optimize(cell, lj_potential_optim)

    assert optimizer.converged, "Gradient Descent Optimizer did not converge"

    energy = lj_potential_optim.calculate_energy(cell)
    forces = cell.get_forces()
    logger.debug(f"Gradient Descent - Post-optimization Energy: {energy:.6f} eV")
    logger.debug(f"Gradient Descent - Post-optimization Forces: {forces}")

    # 检查最大力是否小于容差
    max_force = max(np.linalg.norm(f) for f in forces)
    assert (
        max_force < optimizer.tol
    ), f"Max force {max_force} exceeds tolerance {optimizer.tol}"


def test_bfgs_optimizer(lj_potential_optim, cell_with_multiple_atoms):
    """
    测试 BFGS 优化器，使用4个原子，无周期性边界条件。
    """
    logger = logging.getLogger(__name__)
    optimizer = BFGSOptimizer(tol=1e-4, maxiter=20000)
    cell = cell_with_multiple_atoms

    optimizer.optimize(cell, lj_potential_optim)

    assert optimizer.converged, "BFGS Optimizer did not converge"

    energy = lj_potential_optim.calculate_energy(cell)
    forces = cell.get_forces()
    logger.debug(f"BFGS Optimizer - Post-optimization Energy: {energy:.6f} eV")
    logger.debug(f"BFGS Optimizer - Post-optimization Forces: {forces}")

    # 检查最大力是否小于容差
    max_force = max(np.linalg.norm(f) for f in forces)
    assert (
        max_force < 2 * optimizer.tol
    ), f"Max force {max_force} exceeds tolerance {2 * optimizer.tol}"
