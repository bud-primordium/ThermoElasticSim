import pytest
import numpy as np
from python.structure import Atom, Cell
import logging
from datetime import datetime
import os

# 配置日志记录（已在 conftest.py 中配置，故无需在这里再次配置）


@pytest.fixture
def atom():
    """
    @fixture 创建一个原子实例
    """
    position = np.array([0.0, 0.0, 0.0])
    mass_amu = 26.9815  # amu
    return Atom(id=0, symbol="Al", mass_amu=mass_amu, position=position)


@pytest.fixture
def cell(atom):
    """
    @fixture 创建一个晶胞实例，包含一个原子
    """
    lattice_vectors = np.eye(3) * 4.0  # Å，选择4.0作为简单的例子
    return Cell(lattice_vectors, [atom], pbc_enabled=True)


def test_atom_creation(atom):
    """
    @brief 测试原子的创建
    """
    assert atom.id == 0
    assert atom.symbol == "Al"
    np.testing.assert_array_equal(atom.position, np.array([0.0, 0.0, 0.0]))
    assert atom.mass_amu == 26.9815
    np.testing.assert_array_equal(atom.velocity, np.zeros(3))
    np.testing.assert_array_equal(atom.force, np.zeros(3))


def test_cell_creation(cell, atom):
    """
    @brief 测试晶胞的创建
    """
    np.testing.assert_array_equal(cell.lattice_vectors, np.eye(3) * 4.0)
    assert len(cell.atoms) == 1
    assert cell.atoms[0] == atom
    assert cell.pbc_enabled is True
    assert cell.lattice_locked is False
    assert cell.volume == np.linalg.det(cell.lattice_vectors)


def test_volume_calculation(cell):
    """
    @brief 测试晶胞体积的计算
    """
    expected_volume = np.linalg.det(cell.lattice_vectors)
    calculated_volume = cell.calculate_volume()
    assert np.isclose(
        calculated_volume, expected_volume
    ), "Cell volume calculation is incorrect."


def test_apply_periodic_boundary(cell):
    """
    @brief 测试周期性边界条件的应用
    """
    lattice_constant = cell.lattice_vectors[0, 0]  # 假设晶格为立方
    position = np.array(
        [lattice_constant + 1.0, -1.0, lattice_constant + 1.0]
    )  # 超出晶胞范围的坐标
    new_position = cell.apply_periodic_boundary(position)
    # 检查新位置是否在 [0, lattice_constant) 范围内
    assert np.all(new_position >= 0), "New position has negative components."
    assert np.all(
        new_position < lattice_constant
    ), "New position exceeds lattice constant."


@pytest.mark.parametrize(
    "displacement, expected_disp",
    [
        (np.array([3.0, 3.0, 3.0]), np.array([-1.0, -1.0, -1.0])),
        (np.array([-3.0, -3.0, -3.0]), np.array([1.0, 1.0, 1.0])),
        (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])),
        (np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0])),
        (np.array([4.0, 4.0, 4.0]), np.array([0.0, 0.0, 0.0])),
    ],
)
def test_minimum_image_various(cell, displacement, expected_disp):
    """
    @brief 测试 Cell 类的 minimum_image 方法，使用各种位移向量。
    """
    min_disp = cell.minimum_image(displacement)
    np.testing.assert_array_almost_equal(
        min_disp,
        expected_disp,
        decimal=10,
        err_msg=f"Minimum image displacement for {displacement} is incorrect.",
    )


def test_minimum_image(cell):
    """
    @brief 测试 Cell 类的 minimum_image 方法，确保其正确应用最小镜像原则。
    """
    displacement = np.array([3.0, 3.0, 3.0])  # 示例位移
    min_disp = cell.minimum_image(displacement)
    # lattice_vectors = 4.0 * I, displacement = [3,3,3]
    # fractional = [0.75,0.75,0.75]
    # fractional -= np.round(fractional) = [0.75-1, 0.75-1, 0.75-1] = [-0.25, -0.25, -0.25]
    # min_disp = 4.0 * [-0.25, -0.25, -0.25] = [-1.0, -1.0, -1.0]
    expected_disp = np.array([-1.0, -1.0, -1.0])
    np.testing.assert_array_almost_equal(
        min_disp,
        expected_disp,
        decimal=10,
        err_msg="Minimum image displacement is incorrect.",
    )


def test_invalid_minimum_image(cell):
    """
    @brief 测试 Cell 类的 minimum_image 方法，确保在传入错误形状的位移向量时抛出异常。
    """
    with pytest.raises(ValueError, match="Displacement must be a 3-dimensional vector"):
        cell.minimum_image(np.array([1.0, 2.0]))  # 非3D向量


def test_apply_deformation_locked(cell, atom):
    """
    @brief 测试在锁定晶格向量时施加变形矩阵，仅更新原子位置
    """
    cell.lock_lattice_vectors()
    deformation_matrix = np.array([[1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 1.1]])
    original_position = atom.position.copy()
    cell.apply_deformation(deformation_matrix)
    expected_position = original_position * 1.1
    if cell.pbc_enabled:
        expected_position = cell.apply_periodic_boundary(expected_position)
    np.testing.assert_array_almost_equal(
        atom.position,
        expected_position,
        decimal=10,
        err_msg="Atomic position after deformation is incorrect.",
    )
    # 确保晶格向量未改变
    np.testing.assert_array_equal(
        cell.lattice_vectors,
        np.eye(3) * 4.0,
        err_msg="Lattice vectors should remain unchanged when locked.",
    )


def test_apply_deformation_unlocked(cell, atom):
    """
    @brief 测试在解锁晶格向量时施加变形矩阵，更新晶格和原子位置
    """
    cell.unlock_lattice_vectors()
    deformation_matrix = np.array([[1.1, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.05]])
    original_position = atom.position.copy()

    # 保存原始晶格向量
    original_lattice_vectors = cell.lattice_vectors.copy()

    # 应用变形矩阵
    cell.apply_deformation(deformation_matrix)

    # 计算预期的晶格向量
    expected_lattice_vectors = np.dot(original_lattice_vectors, deformation_matrix.T)

    # 计算原子的位置变换
    fractional = np.linalg.solve(original_lattice_vectors.T, original_position)
    fractional = np.dot(deformation_matrix, fractional)
    expected_position = np.dot(expected_lattice_vectors.T, fractional)

    if cell.pbc_enabled:
        expected_position = cell.apply_periodic_boundary(expected_position)

    # 断言晶格向量是否正确
    np.testing.assert_array_almost_equal(
        cell.lattice_vectors,
        expected_lattice_vectors,
        decimal=10,
        err_msg="Lattice vectors after deformation are incorrect.",
    )

    # 断言原子的位置是否正确
    np.testing.assert_array_almost_equal(
        atom.position,
        expected_position,
        decimal=10,
        err_msg="Atomic position after deformation is incorrect.",
    )


def test_cell_copy(cell):
    """
    @brief 测试 Cell 的深拷贝功能
    """
    cell_copy = cell.copy()
    assert cell_copy is not cell, "Copied cell should be a different instance."
    np.testing.assert_array_equal(cell_copy.lattice_vectors, cell.lattice_vectors)
    assert len(cell_copy.atoms) == len(cell.atoms)
    for original_atom, copied_atom in zip(cell.atoms, cell_copy.atoms):
        assert (
            copied_atom is not original_atom
        ), "Copied atom should be a different instance."
        assert copied_atom.id == original_atom.id
        assert copied_atom.symbol == original_atom.symbol
        assert copied_atom.mass_amu == original_atom.mass_amu
        np.testing.assert_array_equal(copied_atom.position, original_atom.position)
        np.testing.assert_array_equal(copied_atom.velocity, original_atom.velocity)
        np.testing.assert_array_equal(copied_atom.force, original_atom.force)
    assert cell_copy.pbc_enabled == cell.pbc_enabled
    assert cell_copy.lattice_locked == cell.lattice_locked
    assert cell_copy.volume == cell.volume


def test_apply_periodic_boundary_no_pbc():
    """
    @brief 测试在未启用周期性边界条件时，apply_periodic_boundary 方法的行为。
    """
    lattice_vectors = np.eye(3) * 10.0
    atom = Atom(id=0, symbol="Al", mass_amu=26.9815, position=[12.0, -5.0, 15.0])
    cell = Cell(lattice_vectors, [atom], pbc_enabled=False)
    position = np.array([12.0, -5.0, 15.0])
    new_position = cell.apply_periodic_boundary(position)
    # 应用PBC未启用，位置应保持不变
    np.testing.assert_array_equal(
        new_position,
        position,
        err_msg="Position should remain unchanged when PBC is disabled.",
    )
