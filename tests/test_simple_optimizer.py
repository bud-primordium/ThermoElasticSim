# tests/test_simple_optimizer.py

import pytest
import numpy as np
import logging
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.optimizers import GradientDescentOptimizer
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
        f"./logs/test_simple_optimizer_{current_time}.log"  # 生成带时间戳的日志文件名
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
def lj_potential_simple():
    """
    创建一个 Lennard-Jones 势能对象，用于简单系统优化测试。
    """
    return LennardJonesPotential(epsilon=0.0103, sigma=2.55, cutoff=2.5 * 2.55)


@pytest.fixture
def simple_cell():
    """
    创建一个包含两个原子的晶胞，测试优化器的有效性。
    """
    sigma = 2.55
    atoms = [
        Atom(
            id=0,
            symbol="Al",
            mass_amu=26.9815,
            position=[0.0, 0.0, 0.0],
            velocity=None,
        ),
        Atom(
            id=1,
            symbol="Al",
            mass_amu=26.9815,
            position=[3.0, 0.0, 0.0],  # 初始距离为3.0 Å
            velocity=None,
        ),
    ]
    lattice_vectors = np.eye(3) * 10.0  # 盒子大小为10 Å，避免PBC影响
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=False)
    return cell


def test_gradient_descent_optimizer_simple(lj_potential_simple, simple_cell):
    """
    测试梯度下降优化器，使用两个原子，无周期性边界条件。
    """
    logger = logging.getLogger(__name__)
    optimizer = GradientDescentOptimizer(
        max_steps=10000, tol=1e-4, step_size=1e-2, energy_tol=1e-4
    )
    cell = simple_cell

    optimizer.optimize(cell, lj_potential_simple)

    assert (
        optimizer.converged
    ), "Gradient Descent Optimizer did not converge for simple system"

    energy = lj_potential_simple.calculate_energy(cell)
    forces = cell.get_forces()
    logger.debug(f"Simple GD Optimizer - Post-optimization Energy: {energy:.6f} eV")
    logger.debug(f"Simple GD Optimizer - Post-optimization Forces: {forces}")

    # 检查最大力是否小于容差
    max_force = max(np.linalg.norm(f) for f in forces)
    assert (
        max_force < optimizer.tol
    ), f"Max force {max_force} exceeds tolerance {optimizer.tol}"

    # 检查原子间距离是否接近 sigma*2^(1/6)
    expected_distance = 2 ** (1 / 6) * lj_potential_simple.sigma  # 约为2.04 Å
    actual_distance = np.linalg.norm(cell.atoms[1].position - cell.atoms[0].position)
    assert np.isclose(
        actual_distance, expected_distance, atol=1e-2
    ), f"Actual distance {actual_distance} is not close to expected {expected_distance}"
