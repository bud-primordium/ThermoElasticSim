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
    return LennardJonesPotential(
        epsilon=0.0103, sigma=2.55, cutoff=6.375
    )  # cutoff=2.5*sigma


@pytest.fixture
def cell_with_fcc_structure():
    """
    创建一个包含多个原子的面心立方（FCC）晶胞，模拟更复杂的系统（32个原子）。
    """
    sigma = 2.55
    lattice_constant = 5.1  # Å, 选择足够大的晶胞以满足截断半径要求
    num_repetitions = 2  # 在每个方向上复制单位晶胞，生成2x2x2=8单位晶胞，共32个原子
    total_atoms = 4 * num_repetitions**3  # FCC单位晶胞4个原子

    # 定义单位FCC晶胞的原子位置（基于单位晶胞）
    unit_cell_atoms = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]

    atoms = []
    for i in range(num_repetitions):
        for j in range(num_repetitions):
            for k in range(num_repetitions):
                for atom_pos in unit_cell_atoms:
                    pos = np.array(
                        [
                            (i + atom_pos[0]) * lattice_constant,
                            (j + atom_pos[1]) * lattice_constant,
                            (k + atom_pos[2]) * lattice_constant,
                        ]
                    )
                    # 添加微小扰动以打破完美对称性
                    perturbation = np.random.uniform(
                        -0.01, 0.01, size=3
                    )  # 0.01 Å的随机扰动
                    pos += perturbation
                    atoms.append(
                        Atom(
                            id=len(atoms),
                            symbol="Al",
                            mass_amu=26.9815,
                            position=pos,
                            velocity=None,
                        )
                    )

    # 定义晶格矢量
    lattice_vectors = np.eye(3) * lattice_constant * num_repetitions

    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    return cell


def test_gradient_descent_optimizer(lj_potential_optim, cell_with_fcc_structure):
    """
    测试梯度下降优化器，使用32个原子，面心立方结构。
    """
    logger = logging.getLogger(__name__)
    optimizer = GradientDescentOptimizer(
        max_steps=20000, tol=1e-3, step_size=1e-3, energy_tol=1e-4
    )
    cell = cell_with_fcc_structure

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


def test_bfgs_optimizer(lj_potential_optim, cell_with_fcc_structure):
    """
    测试 BFGS 优化器，使用32个原子，面心立方结构。
    """
    logger = logging.getLogger(__name__)
    optimizer = BFGSOptimizer(tol=1e-4, maxiter=20000)
    cell = cell_with_fcc_structure

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
