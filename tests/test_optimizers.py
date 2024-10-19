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
    log_filename = f"./logs/optimizers/test_optimizers_{current_time}.log"  # 生成带时间戳的日志文件名

    # 创建文件处理器
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件日志级别
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    yield

    # 测试结束后移除处理器
    logger.removeHandler(ch)
    logger.removeHandler(fh)


def generate_fcc_positions(lattice_constant, repetitions):
    """
    生成面心立方 (FCC) 结构的原子位置。

    @param lattice_constant 晶格常数，单位 Å
    @param repetitions 每个方向上的单位晶胞重复次数
    @return 原子位置列表
    """
    # FCC 单位晶胞的标准原子位置（分数坐标）
    unit_cell_positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]

    positions = []
    for i in range(repetitions):
        for j in range(repetitions):
            for k in range(repetitions):
                for pos in unit_cell_positions:
                    cartesian = (
                        np.array(pos) * lattice_constant
                        + np.array([i, j, k]) * lattice_constant
                    )
                    positions.append(cartesian)

    return positions


@pytest.fixture
def lj_potential_optim():
    """
    创建一个 Lennard-Jones 势能对象，用于优化测试。
    """
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma  # 截断半径，确保小于盒子的一半
    return LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)


@pytest.fixture
def fcc_cell():
    """
    创建一个包含多个原子的面心立方 (FCC) 晶胞，用于优化测试。

    @return Cell 实例
    """
    lattice_constant = 15.0  # Å
    repetitions = 1  # 每个方向上的单位晶胞重复次数，增加以获得更多原子

    # 生成 FCC 结构的原子位置
    positions = generate_fcc_positions(lattice_constant, repetitions)

    # 创建 Atom 实例列表
    atoms = []
    for idx, pos in enumerate(positions):
        atoms.append(
            Atom(
                id=idx,
                symbol="Al",
                mass_amu=26.9815,
                position=np.array(pos),
                velocity=None,
            )
        )

    # 定义晶格矢量
    lattice_vectors = np.eye(3) * lattice_constant  # 单位晶胞大小

    # 创建 Cell 实例
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    return cell


def test_gradient_descent_optimizer(lj_potential_optim, fcc_cell):
    """
    测试梯度下降优化器，使用面心立方 (FCC) 结构。
    """
    logger = logging.getLogger(__name__)
    optimizer = GradientDescentOptimizer(
        max_steps=20000, tol=1e-3, step_size=1e-3, energy_tol=1e-4
    )
    cell = fcc_cell.copy()  # 使用深拷贝以避免修改原始晶胞

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


def test_bfgs_optimizer(lj_potential_optim, fcc_cell):
    """
    测试 BFGS 优化器，使用面心立方 (FCC) 结构。
    """
    logger = logging.getLogger(__name__)
    optimizer = BFGSOptimizer(tol=1e-4, maxiter=20000)
    cell = fcc_cell.copy()  # 使用深拷贝以避免修改原始晶胞

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
