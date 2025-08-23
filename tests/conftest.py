"""
pytest配置文件 - 提供全局fixtures和测试配置
"""

import numpy as np
import pytest

from thermoelasticsim.core.structure import Atom, Cell


@pytest.fixture
def sample_atom():
    """创建一个标准的氢原子用于测试"""
    return Atom(
        id=1,
        symbol="H",
        mass_amu=1.008,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.1, 0.2, 0.3]),
    )


@pytest.fixture
def sample_atoms():
    """创建多个原子用于测试"""
    atoms = [
        Atom(id=0, symbol="H", mass_amu=1.008, position=[0.0, 0.0, 0.0]),
        Atom(id=1, symbol="O", mass_amu=15.999, position=[1.0, 0.0, 0.0]),
        Atom(id=2, symbol="H", mass_amu=1.008, position=[1.5, 0.5, 0.0]),
    ]
    return atoms


@pytest.fixture
def simple_lattice():
    """创建简单的立方晶格"""
    return np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])


@pytest.fixture
def sample_cell(simple_lattice, sample_atoms):
    """创建一个标准的晶胞用于测试"""
    return Cell(lattice_vectors=simple_lattice, atoms=sample_atoms, pbc_enabled=True)


@pytest.fixture
def orthorhombic_lattice():
    """创建正交晶格"""
    return np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])


@pytest.fixture
def triclinic_lattice():
    """创建三斜晶格用于测试复杂情况"""
    return np.array([[3.0, 0.0, 0.0], [1.0, 2.5, 0.0], [0.5, 0.5, 4.0]])


@pytest.fixture
def deformation_matrices():
    """提供各种变形矩阵用于测试"""
    return {
        "identity": np.eye(3),
        "small_strain": np.array([[1.01, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        "shear": np.array([[1.0, 0.05, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        "uniform_compression": np.array(
            [[0.95, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.95]]
        ),
    }


# 全局测试配置
def pytest_configure(config):
    """pytest全局配置"""
    # 设置numpy错误处理
    np.seterr(all="raise")


def pytest_runtest_setup(item):
    """每个测试前的设置"""
    # 设置随机种子确保可重现性
    np.random.seed(42)
