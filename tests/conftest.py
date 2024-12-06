# 文件名: tests/conftest.py
# 作者: Gilbert Young
# 修改日期: 2024-10-31
# 文件描述: 定义共享的 pytest fixture，供多个测试文件使用。

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

#################################################
# 日志配置相关 fixtures
#################################################


@pytest.fixture(scope="session", autouse=True)
def configure_root_logging():
    """配置根日志记录器，确保日志的基本配置"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    yield

    logger.removeHandler(ch)


@pytest.fixture(scope="module", autouse=True)
def configure_module_logging(request):
    """配置每个测试模块的日志记录，将日志保存到各自的子目录"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    module_path = request.node.nodeid.split("::")[0]
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    log_directory = f"./logs/{module_name}/"
    os.makedirs(log_directory, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_directory}/{module_name}_{current_time}.log"

    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    yield

    logger.removeHandler(fh)


#################################################
# 原子和晶胞相关的基础 fixtures
#################################################


@pytest.fixture
def atom():
    """创建一个基本的原子实例"""
    position = np.array([0.0, 0.0, 0.0])
    mass_amu = 26.9815  # amu
    return Atom(id=0, symbol="Al", mass_amu=mass_amu, position=position)


@pytest.fixture
def two_atom_cell():
    """创建一个包含两个原子的晶胞实例"""
    lattice_vectors = np.eye(3) * 5.1  # Å
    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0]),
        Atom(id=1, symbol="Al", mass_amu=26.9815, position=[2.55, 0.0, 0.0]),
    ]
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    return cell


@pytest.fixture
def simple_cell():
    """创建一个简单的两原子晶胞"""
    lattice_vectors = np.eye(3) * 6.0  # Å
    mass_amu = 26.9815  # amu (Aluminum)
    position1 = np.array([0.0, 0.0, 0.0])
    position2 = np.array([2.55, 0.0, 0.0])
    atom1 = Atom(id=0, symbol="Al", mass_amu=mass_amu, position=position1)
    atom2 = Atom(id=1, symbol="Al", mass_amu=mass_amu, position=position2)
    cell = Cell(lattice_vectors=lattice_vectors, atoms=[atom1, atom2], pbc_enabled=True)
    return cell


#################################################
# FCC结构相关 fixtures
#################################################


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

    # 验证生成的原子数量
    expected_num_atoms = 4 * repetitions**3
    assert (
        len(positions) == expected_num_atoms
    ), f"Generated {len(positions)} positions, expected {expected_num_atoms}"

    return positions


@pytest.fixture
def fcc_cell():
    """创建一个标准的FCC晶胞实例"""
    lattice_constant = 4.05  # Å
    repetitions = 2
    positions = generate_fcc_positions(lattice_constant, repetitions)

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

    lattice_vectors = np.eye(3) * lattice_constant * repetitions
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    return cell


@pytest.fixture
def large_fcc_cell():
    """创建一个较大的FCC铝晶胞，用于弹性常数计算"""
    lattice_constant = 4.05  # Å (铝的实验晶格常数)
    repetitions = 3  # 3x3x3 超胞
    positions = generate_fcc_positions(lattice_constant, repetitions)

    atoms = []
    for idx, pos in enumerate(positions):
        atoms.append(
            Atom(id=idx, symbol="Al", mass_amu=26.9815, position=np.array(pos))
        )

    lattice_vectors = np.eye(3) * lattice_constant * repetitions
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    return cell


#################################################
# 势能和邻居列表相关 fixtures
#################################################


@pytest.fixture
def lj_potential():
    """创建一个基本的 Lennard-Jones 势能对象"""
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    return LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)


@pytest.fixture
def lj_potential_with_neighbor_list(simple_cell, lj_potential):
    """为简单晶胞创建带邻居列表的 Lennard-Jones 势能对象"""
    neighbor_list = NeighborList(cutoff=lj_potential.cutoff)
    neighbor_list.build(simple_cell)
    lj_potential.set_neighbor_list(neighbor_list)
    return lj_potential


@pytest.fixture
def lj_potential_for_fcc():
    """创建一个适用于 FCC 结构的 Lennard-Jones 势能对象"""
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma  # Å
    return LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)


@pytest.fixture
def lj_potential_with_neighbor_list_fcc(large_fcc_cell, lj_potential_for_fcc):
    """创建适用于 FCC 结构的带邻居列表的 Lennard-Jones 势能对象"""
    neighbor_list = NeighborList(cutoff=lj_potential_for_fcc.cutoff)
    neighbor_list.build(large_fcc_cell)
    lj_potential_for_fcc.set_neighbor_list(neighbor_list)
    return lj_potential_for_fcc


@pytest.fixture
def two_atom_neighbor_list(two_atom_cell, lj_potential):
    """创建两原子系统的邻居列表"""
    neighbor_list = NeighborList(cutoff=lj_potential.cutoff)
    neighbor_list.build(two_atom_cell)
    lj_potential.set_neighbor_list(neighbor_list)
    return neighbor_list


#################################################
# 接口相关 fixtures
#################################################


@pytest.fixture
def lj_interface():
    """创建 Lennard-Jones C++ 接口实例"""
    return CppInterface("lennard_jones")


#################################################
# 字体相关 fixtures
#################################################
@pytest.fixture(scope="session", autouse=True)
def configure_matplotlib():
    """配置 matplotlib 的全局设置"""
    from matplotlib import rcParams
    import matplotlib.pyplot as plt

    # 设置默认字体和日志级别
    rcParams["font.family"] = "DejaVu Sans"
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # 关闭交互模式
    plt.ioff()
