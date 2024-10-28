# 文件名: tests/test_pbc.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 测试周期性边界条件 (PBC) 的应用和变形方法的正确性。

import pytest
import numpy as np
from python.structure import Atom, Cell
import logging


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
    position = np.array([6.0, 6.0, 6.0]).reshape(3, 1)  # 调整形状为 (3, 1)
    new_position = cell.apply_periodic_boundary(position).flatten()
    expected_position = np.array([1.0, 1.0, 1.0])  # 6 % 5 = 1
    assert np.allclose(
        new_position, expected_position
    ), f"Expected {expected_position}, got {new_position}"
    logger.debug(f"PBC applied correctly: {new_position} == {expected_position}")

    # 测试未启用 PBC
    cell_no_pbc = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=False)
    new_position_no_pbc = cell_no_pbc.apply_periodic_boundary(position).flatten()
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
    lattice_vectors = np.eye(3) * 2.55  # 盒子长度为 2.55 Å
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 定义一个变形矩阵（包括缩放和剪切）
    deformation_matrix = np.array(
        [
            [1.001, 0.01, 0.0],
            [0.01, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # 施加变形
    cell.apply_deformation(deformation_matrix)

    # 变形后的晶格向量应被正确更新
    expected_lattice_vectors = np.dot(np.eye(3) * 2.55, deformation_matrix)
    assert np.allclose(
        cell.lattice_vectors, expected_lattice_vectors, atol=1e-5
    ), f"Expected lattice vectors {expected_lattice_vectors}, got {cell.lattice_vectors}"
    logger.debug(f"Deformed lattice vectors correctly:\n{cell.lattice_vectors}")

    # 原子0: [0,0,0] 应保持不变
    assert np.allclose(
        cell.atoms[0].position, [0.0, 0.0, 0.0], atol=1e-5
    ), f"Expected atom0 position [0.0, 0.0, 0.0], got {cell.atoms[0].position}"

    # 原子1: 通过变形矩阵进行线性变换，并应用 PBC
    deformed_position = np.dot([2.55, 2.55, 2.55], deformation_matrix)
    logger.debug(f"Deformed position before PBC: {deformed_position}")

    # 先计算分数坐标
    fractional_coords = np.linalg.solve(expected_lattice_vectors.T, deformed_position)
    logger.debug(f"Fractional coordinates before PBC: {fractional_coords}")

    fractional_coords %= 1.0  # 应用周期性边界
    logger.debug(f"Fractional coordinates after PBC: {fractional_coords}")

    # 转换回笛卡尔坐标
    expected_atom1_position_pbc = np.dot(
        expected_lattice_vectors.T, fractional_coords
    ).flatten()
    logger.debug(f"Expected atom 1 position after PBC: {expected_atom1_position_pbc}")

    # 实际位置
    actual_position = cell.atoms[1].position
    logger.debug(f"Actual atom 1 position: {actual_position}")

    assert np.allclose(
        actual_position, expected_atom1_position_pbc, atol=1e-4
    ), f"Expected atom1 position {expected_atom1_position_pbc}, got {actual_position}"
