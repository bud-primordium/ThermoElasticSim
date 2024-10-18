# tests/test_lj.py

import pytest
import numpy as np
import logging
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential


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
    fh = logging.FileHandler("test_potential.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件日志级别
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    yield
    # 测试结束后，可以在这里添加清理代码（如果需要）


def test_lj_potential_at_r_m():
    """
    @brief 测试 Lennard-Jones 势在 r = r_m (2^(1/6)*sigma) 时的势能和力。
    """
    logger = logging.getLogger(__name__)
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    # 计算 r_m = 2^(1/6) * sigma
    r_m = 2 ** (1 / 6) * sigma

    # 创建两个原子，距离为 r_m
    atoms = [
        Atom(
            id=0,
            mass_amu=26.9815,
            position=np.array([0.0, 0.0, 0.0]),
            symbol="Al",
        ),
        Atom(
            id=1,
            mass_amu=26.9815,
            position=np.array([r_m, 0.0, 0.0]),
            symbol="Al",
        ),
    ]
    lattice_vectors = np.eye(3) * (sigma * 3)  # 确保盒子足够大
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 计算势能和力
    energy = lj_potential.calculate_energy(cell)
    lj_potential.calculate_forces(cell)
    forces = cell.get_forces()

    # 断言
    assert np.isclose(
        energy, -epsilon, atol=1e-6
    ), f"Energy at r_m should be -epsilon, got {energy}"
    assert np.allclose(
        forces, 0.0, atol=1e-6
    ), f"Forces at r_m should be zero, got {forces}"
    logger.debug(f"LJ Potential at r_m: Energy = {energy} eV, Forces = {forces}")
