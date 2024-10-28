# tests/conftest.py

import pytest
import numpy as np
import logging
import os
from datetime import datetime
from itertools import product
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.interfaces.cpp_interface import CppInterface
from python.utils import NeighborList


@pytest.fixture(scope="session", autouse=True)
def configure_root_logging():
    """
    配置根日志记录器，确保日志的基本配置。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # 控制台日志级别
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    yield

    # 测试结束后移除控制台处理器
    logger.removeHandler(ch)


@pytest.fixture(scope="module", autouse=True)
def configure_module_logging(request):
    """
    配置每个测试模块的日志记录，将日志保存到各自的子目录。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 获取测试模块的名称
    module_path = request.node.nodeid.split("::")[0]  # e.g., "tests/test_potentials.py"
    module_name = os.path.splitext(os.path.basename(module_path))[
        0
    ]  # e.g., "test_potentials"

    # 创建日志目录
    log_directory = f"./logs/{module_name}/"
    os.makedirs(log_directory, exist_ok=True)

    # 生成带时间戳的日志文件名
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_directory}/{module_name}_{current_time}.log"

    # 创建文件处理器
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件日志级别
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    yield

    # 测试模块结束后移除文件处理器
    logger.removeHandler(fh)


@pytest.fixture
def lj_potential():
    """
    创建一个 Lennard-Jones 势能对象。
    """
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    return LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)


@pytest.fixture
def lj_interface():
    """
    创建一个 C++ 接口实例，用于 Lennard-Jones 势能计算。
    """
    return CppInterface("lennard_jones")


@pytest.fixture
def two_atom_cell():
    """
    创建一个包含两个原子的晶胞实例。
    """
    lattice_vectors = np.eye(3) * 5.1  # Å，确保盒子足够大以包含两个原子
    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0]),
        Atom(
            id=1, symbol="Al", mass_amu=26.9815, position=[2.55, 0.0, 0.0]
        ),  # r = sigma = 2.55 Å
    ]
    cell = Cell(lattice_vectors, atoms, pbc_enabled=True)
    return cell


@pytest.fixture
def two_atom_neighbor_list(two_atom_cell):
    """
    创建并返回两个原子的邻居列表。
    """
    cutoff = 2.5 * 2.55  # 2.5 * sigma
    neighbor_list = NeighborList(cutoff=cutoff)
    neighbor_list.build(two_atom_cell)
    return neighbor_list


def generate_fcc_positions(lattice_constant, repetitions):
    """
    生成面心立方 (FCC) 结构的原子位置。

    Parameters
    ----------
    lattice_constant : float
        FCC 晶格常数，单位 Å。
    repetitions : int
        每个方向上的单位晶胞重复次数。

    Returns
    -------
    list of list of float
        原子位置列表。
    """
    # FCC 单位晶胞的标准原子位置（分数坐标）
    base_positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]

    positions = []
    for i, j, k in product(range(repetitions), repeat=3):
        for pos in base_positions:
            cartesian = (np.array([i, j, k]) + np.array(pos)) * lattice_constant
            positions.append(cartesian.tolist())

    return positions
