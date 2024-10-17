# tests/test_deformation.py

import pytest
import numpy as np
from python.deformation import Deformer


def test_generate_deformation_matrices():
    """
    @brief 测试生成变形矩阵的方法
    """
    delta = 0.01  # 1% 的应变
    deformer = Deformer(delta)
    deformation_matrices = deformer.generate_deformation_matrices()

    # 检查是否生成了 6 个变形矩阵
    assert len(deformation_matrices) == 6

    # 检查每个变形矩阵的正确性
    for i, F in enumerate(deformation_matrices):
        # 检查矩阵是否为 3x3
        assert F.shape == (3, 3)

        # 检查矩阵是否为单位矩阵加上微小变形
        expected_F = np.identity(3)
        if i < 3:
            expected_F[i, i] += delta
        else:
            shear_indices = [(0, 1), (0, 2), (1, 2)]
            idx = i - 3
            i_shear, j_shear = shear_indices[idx]
            expected_F[i_shear, j_shear] += delta / 2
            expected_F[j_shear, i_shear] += delta / 2
        np.testing.assert_array_almost_equal(F, expected_F, decimal=6)


def test_apply_deformation(cell_fixture):
    """
    @brief 测试对晶胞施加变形矩阵的方法
    """
    delta = 0.01
    deformer = Deformer(delta)
    deformation_matrix = deformer.generate_deformation_matrices()[0]  # ε_xx
    original_lattice = cell_fixture.lattice_vectors.copy()
    original_positions = [atom.position.copy() for atom in cell_fixture.atoms]

    deformer.apply_deformation(cell_fixture, deformation_matrix)

    # 检查晶格矢量是否正确更新
    expected_lattice = np.dot(original_lattice, deformation_matrix.T)
    np.testing.assert_array_almost_equal(
        cell_fixture.lattice_vectors, expected_lattice, decimal=6
    )

    # 检查原子位置是否正确更新
    for original_pos, atom in zip(original_positions, cell_fixture.atoms):
        expected_pos = np.dot(deformation_matrix, original_pos)
        np.testing.assert_array_almost_equal(atom.position, expected_pos, decimal=6)


@pytest.fixture
def cell_fixture():
    """
    @fixture 创建一个简单的晶胞，用于测试
    """
    from python.structure import Atom, Cell

    mass = 26.9815  # amu
    position = np.array([0.0, 0.0, 0.0])
    atom = Atom(id=0, symbol="Al", mass=mass, position=position)
    lattice_vectors = np.eye(3) * 4.05  # Å
    cell = Cell(lattice_vectors, [atom], pbc_enabled=True)
    return cell
