import pytest
import numpy as np
from python.structure import Atom, Cell

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
    expected_position = np.dot(cell.lattice_vectors.T, fractional)

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


# 新增针对 build_supercell 方法的测试
def test_build_supercell(cell):
    """
    @brief 测试 build_supercell 方法，确保超胞构建正确，且没有原子重叠
    """
    repetition = (2, 2, 2)
    super_cell = cell.build_supercell(repetition)

    # 检查超胞的晶格矢量是否正确
    expected_lattice_vectors = np.dot(np.diag(repetition), cell.lattice_vectors)
    np.testing.assert_array_equal(
        super_cell.lattice_vectors,
        expected_lattice_vectors,
        err_msg="Supercell lattice vectors are incorrect.",
    )

    # 计算预期的原子数量
    expected_num_atoms = cell.num_atoms * np.prod(repetition)
    assert (
        super_cell.num_atoms == expected_num_atoms
    ), "Supercell has incorrect number of atoms."

    # 检查原子位置是否正确，且没有原子重叠
    positions = np.array([atom.position for atom in super_cell.atoms])
    # 计算原子间距矩阵
    distance_matrix = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
    )
    # 设置对角线为无穷大，避免自我比较
    np.fill_diagonal(distance_matrix, np.inf)
    min_distance = np.min(distance_matrix)
    # 检查最小原子间距是否大于某个阈值，避免原子重叠（例如 0.1 Å）
    assert (
        min_distance > 0.1
    ), "Atoms in supercell are too close, possible overlap detected."


def test_build_supercell_no_overlap(cell):
    """
    @brief 测试构建的超胞在周期性边界条件下没有原子重叠
    """
    repetition = (2, 2, 2)
    super_cell = cell.build_supercell(repetition)

    # 获取原子位置并转换到分数坐标
    positions = np.array([atom.position for atom in super_cell.atoms])
    fractional = np.dot(positions, np.linalg.inv(super_cell.lattice_vectors.T))

    # 将分数坐标映射到 [0, 1) 区间
    fractional %= 1.0

    # 检查是否有重复的原子分数坐标
    unique_fractional = np.unique(fractional, axis=0)
    assert len(unique_fractional) == len(
        fractional
    ), "Duplicate fractional coordinates detected in supercell."


def test_build_supercell_pbc(cell):
    """
    @brief 测试构建的超胞是否正确处理周期性边界条件
    """
    repetition = (2, 2, 2)
    super_cell = cell.build_supercell(repetition)

    # 获取超胞中原子的分数坐标
    positions = np.array([atom.position for atom in super_cell.atoms])
    fractional = np.dot(positions, np.linalg.inv(super_cell.lattice_vectors.T))

    # 检查分数坐标是否在 [0, nx), [0, ny), [0, nz) 范围内
    nx, ny, nz = repetition
    assert np.all(fractional[:, 0] >= 0) and np.all(
        fractional[:, 0] < nx
    ), "Fractional x-coordinates out of bounds."
    assert np.all(fractional[:, 1] >= 0) and np.all(
        fractional[:, 1] < ny
    ), "Fractional y-coordinates out of bounds."
    assert np.all(fractional[:, 2] >= 0) and np.all(
        fractional[:, 2] < nz
    ), "Fractional z-coordinates out of bounds."

    # 将分数坐标映射到 [0, 1) 区间
    fractional_mod = fractional % 1.0

    # 检查映射后的分数坐标是否正确处理了 PBC
    assert np.all(
        (fractional_mod >= 0) & (fractional_mod < 1)
    ), "Fractional coordinates after PBC application are out of bounds."


def test_build_supercell_large(cell):
    """
    @brief 测试更大尺寸的超胞构建，确保在更大规模下功能正常
    """
    repetition = (3, 3, 3)
    super_cell = cell.build_supercell(repetition)

    # 检查超胞的晶格矢量是否正确
    expected_lattice_vectors = np.dot(np.diag(repetition), cell.lattice_vectors)
    np.testing.assert_array_equal(
        super_cell.lattice_vectors,
        expected_lattice_vectors,
        err_msg="Supercell lattice vectors are incorrect for large supercell.",
    )

    # 检查原子数量
    expected_num_atoms = cell.num_atoms * np.prod(repetition)
    assert (
        super_cell.num_atoms == expected_num_atoms
    ), "Large supercell has incorrect number of atoms."

    # 检查是否有原子重叠
    positions = np.array([atom.position for atom in super_cell.atoms])
    distance_matrix = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
    )
    np.fill_diagonal(distance_matrix, np.inf)
    min_distance = np.min(distance_matrix)
    assert (
        min_distance > 0.1
    ), "Atoms in large supercell are too close, possible overlap detected."
