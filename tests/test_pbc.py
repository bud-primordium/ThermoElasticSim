# tests/test_pbc.py

import pytest
import numpy as np
from python.structure import Atom, Cell
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
