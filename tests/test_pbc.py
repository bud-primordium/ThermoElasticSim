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
    测试 Cell.apply_periodic_boundary 方法是否正确应用周期性边界条件
    """
    logger = logging.getLogger(__name__)

    atoms = [
        Atom(
            id=0,
            symbol="Al",
            mass_amu=26.9815,
            position=np.array([6.0, 6.0, 6.0]),
            velocity=None,
        ),
    ]
    lattice_vectors = np.eye(3) * 5.0  # 盒子长度为 5 Å
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 测试单个位置的情况
    position = np.array([6.0, 6.0, 6.0])  # 现在直接使用1D数组
    new_position = cell.apply_periodic_boundary(position)
    expected_position = np.array([1.0, 1.0, 1.0])  # 6 % 5 = 1
    assert np.allclose(
        new_position, expected_position
    ), f"Expected {expected_position}, got {new_position}"
    logger.debug(f"PBC applied correctly: {new_position} == {expected_position}")

    # 测试多个位置的情况
    positions = np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
    new_positions = cell.apply_periodic_boundary(positions)
    expected_positions = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert np.allclose(
        new_positions, expected_positions
    ), f"Expected {expected_positions}, got {new_positions}"

    # 测试未启用 PBC
    cell_no_pbc = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=False)
    new_position_no_pbc = cell_no_pbc.apply_periodic_boundary(position)
    expected_position_no_pbc = np.array([6.0, 6.0, 6.0])
    assert np.allclose(
        new_position_no_pbc, expected_position_no_pbc
    ), f"Expected {expected_position_no_pbc}, got {new_position_no_pbc}"


def test_apply_deformation_with_pbc():
    """
    测试 Cell.apply_deformation 方法在启用 PBC 时是否正确应用变形和 PBC
    """
    logger = logging.getLogger(__name__)

    atoms = [
        Atom(
            id=0,
            symbol="Al",
            mass_amu=26.9815,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=None,
        ),
        Atom(
            id=1,
            symbol="Al",
            mass_amu=26.9815,
            position=np.array([2.55, 2.55, 2.55]),
            velocity=None,
        ),
    ]
    lattice_vectors = np.eye(3) * 2.55
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    deformation_matrix = np.array(
        [
            [1.001, 0.01, 0.0],
            [0.01, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cell.apply_deformation(deformation_matrix)

    # 验证晶格向量更新
    expected_lattice_vectors = np.dot(np.eye(3) * 2.55, deformation_matrix.T)
    assert np.allclose(cell.lattice_vectors, expected_lattice_vectors, atol=1e-5)

    # 验证原子位置更新
    assert np.allclose(cell.atoms[0].position, [0.0, 0.0, 0.0], atol=1e-5)

    # 计算期望的原子1位置
    deformed_position = np.dot([2.55, 2.55, 2.55], deformation_matrix)
    fractional_coords = np.dot(deformed_position, cell.lattice_inv)
    fractional_coords %= 1.0
    expected_atom1_position = np.dot(fractional_coords, cell.lattice_vectors.T)

    assert np.allclose(cell.atoms[1].position, expected_atom1_position, atol=1e-4)
