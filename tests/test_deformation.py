# tests/test_deformation.py

import pytest
import numpy as np
from python.deformation import Deformer


def test_generate_deformation_matrices():
    """
    测试生成变形矩阵的方法，
    考虑到剪切应变的特殊处理
    """
    delta = 0.01  # 1% 的应变
    num_steps = 5  # 设置步数
    deformer = Deformer(delta, num_steps=num_steps)
    deformation_matrices = deformer.generate_deformation_matrices()

    # 检查矩阵数量
    expected_matrices = 6 * num_steps
    assert (
        len(deformation_matrices) == expected_matrices
    ), f"Expected {expected_matrices} matrices, got {len(deformation_matrices)}"

    # 检查应变范围
    strain_values = np.linspace(-delta, delta, num_steps)

    # 遍历每种变形类型
    for i in range(6):
        # 检查该变形类型的所有步长
        for j, strain in enumerate(strain_values):
            matrix_idx = i * num_steps + j
            F = deformation_matrices[matrix_idx]

            # 检查矩阵是否为 3x3
            assert F.shape == (3, 3)

            # 构造期望的变形矩阵
            expected_F = np.identity(3)
            if i < 3:  # 正应变
                expected_F[i, i] += strain
            else:  # 剪切应变
                shear_indices = [(0, 1), (0, 2), (1, 2)]
                idx = i - 3
                i_shear, j_shear = shear_indices[idx]
                # 剪切应变直接使用完整的应变值，不需要除以2
                expected_F[i_shear, j_shear] += strain
                expected_F[j_shear, i_shear] += strain

            np.testing.assert_array_almost_equal(
                F,
                expected_F,
                decimal=6,
                err_msg=f"Mismatch at deformation type {i}, step {j}",
            )


def test_apply_deformation(simple_cell):  # 使用 simple_cell 替代 cell_fixture
    """
    测试对晶胞施加变形矩阵的方法
    """
    delta = 0.01
    num_steps = 5
    deformer = Deformer(delta, num_steps=num_steps)
    deformation_matrices = deformer.generate_deformation_matrices()

    # 选择第一种变形的中间步长进行测试
    test_matrix_idx = num_steps // 2
    deformation_matrix = deformation_matrices[test_matrix_idx]

    # 保存初始状态
    original_lattice = simple_cell.lattice_vectors.copy()
    original_positions = np.array([atom.position.copy() for atom in simple_cell.atoms])

    # 施加变形
    deformer.apply_deformation(simple_cell, deformation_matrix)

    # 验证晶格矢量更新
    expected_lattice = np.dot(original_lattice, deformation_matrix.T)
    np.testing.assert_array_almost_equal(
        simple_cell.lattice_vectors,
        expected_lattice,
        decimal=6,
        err_msg="Lattice vectors not updated correctly",
    )

    # 验证原子位置更新
    for i, (original_pos, atom) in enumerate(
        zip(original_positions, simple_cell.atoms)
    ):
        expected_pos = np.dot(deformation_matrix, original_pos)
        if simple_cell.pbc_enabled:
            expected_pos = simple_cell.apply_periodic_boundary(expected_pos)

        np.testing.assert_array_almost_equal(
            atom.position,
            expected_pos,
            decimal=6,
            err_msg=f"Atom {i} position not updated correctly",
        )
