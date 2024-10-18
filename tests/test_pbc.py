# tests/test_pbc.py

import pytest
import numpy as np
from python.structure import Atom, Cell
import logging
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
    log_filename = f"test_pbc_{current_time}.log"  # 生成带时间戳的日志文件名

    # 创建文件处理器
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件日志级别
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    yield

    # 测试结束后移除处理器
    logger.removeHandler(ch)
    logger.removeHandler(fh)


def test_apply_periodic_boundary():
    """
    @brief 测试 Cell.apply_periodic_boundary 方法是否正确应用周期性边界条件。
    """
    logger = logging.getLogger(__name__)

    atoms = [
        Atom(
            id=0, symbol="Al", mass_amu=26.9815, position=[6.0, 6.0, 6.0], velocity=None
        ),
    ]
    lattice_vectors = np.eye(3) * 5.0  # 盒子长度为 5 Å
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 应用 PBC
    new_position = cell.apply_periodic_boundary(atoms[0].position)
    expected_position = np.array([1.0, 1.0, 1.0])  # 6 % 5 = 1
    assert np.allclose(
        new_position, expected_position
    ), f"Expected {expected_position}, got {new_position}"
    logger.debug(f"PBC applied correctly: {new_position} == {expected_position}")

    # 测试未启用 PBC
    cell_no_pbc = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=False)
    new_position_no_pbc = cell_no_pbc.apply_periodic_boundary(atoms[0].position)
    expected_position_no_pbc = np.array([6.0, 6.0, 6.0])
    assert np.allclose(
        new_position_no_pbc, expected_position_no_pbc
    ), f"Expected {expected_position_no_pbc}, got {new_position_no_pbc}"
    logger.debug(
        f"PBC not applied correctly: {new_position_no_pbc} == {expected_position_no_pbc}"
    )


def test_apply_deformation_with_pbc():
    """
    @brief 测试 Cell.apply_deformation 方法在启用 PBC 时是否正确应用变形和 PBC。
    """
    logger = logging.getLogger(__name__)

    # 创建两个原子，确保变形后 PBC 能正确处理
    atoms = [
        Atom(
            id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0], velocity=None
        ),
        Atom(
            id=1,
            symbol="Al",
            mass_amu=26.9815,
            position=[2.55, 2.55, 2.55],
            velocity=None,
        ),  # a/sqrt(2) for FCC
    ]
    lattice_vectors = np.eye(3) * 5.1  # 盒子长度为 5.1 Å
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 定义一个简单的变形矩阵（缩放）
    deformation_matrix = np.array(
        [
            [1.001, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # 施加变形
    cell.apply_deformation(deformation_matrix)

    # 变形后的晶格向量应被正确更新
    expected_lattice_vectors = np.eye(3) * 5.1
    expected_lattice_vectors[0, 0] *= 1.001  # x方向拉伸
    assert np.allclose(
        cell.lattice_vectors, expected_lattice_vectors, atol=1e-5
    ), f"Expected lattice vectors {expected_lattice_vectors}, got {cell.lattice_vectors}"
    logger.debug(f"Deformed lattice vectors correctly:\n{cell.lattice_vectors}")

    # 变形后的原子位置应被正确更新并应用 PBC
    # 原子0: [0,0,0] 应保持不变
    assert np.allclose(
        cell.atoms[0].position, [0.0, 0.0, 0.0], atol=1e-5
    ), f"Expected atom0 position [0.0, 0.0, 0.0], got {cell.atoms[0].position}"
    # 原子1: [2.55,2.55,2.55] -> [2.55*1.001,2.55,2.55] = [2.55255, 2.55, 2.55]
    expected_atom1_position = np.array([2.55255, 2.55, 2.55])
    assert np.allclose(
        cell.atoms[1].position, expected_atom1_position, atol=1e-3
    ), f"Expected atom1 position {expected_atom1_position}, got {cell.atoms[1].position}"
    logger.debug(
        f"Deformed atom positions correctly: {cell.atoms[1].position} == {expected_atom1_position}"
    )
