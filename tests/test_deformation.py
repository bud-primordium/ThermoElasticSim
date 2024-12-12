# tests/test_deformation.py

import pytest
import numpy as np
from python.deformation import Deformer
from python.utils import TensorConverter


def test_generate_deformation_matrices():
    delta = 0.01  # 1% 应变
    num_steps = 5
    deformer = Deformer(delta, num_steps=num_steps)
    F_list = deformer.generate_deformation_matrices()

    expected_matrices = 6 * num_steps
    assert (
        len(F_list) == expected_matrices
    ), f"Expected {expected_matrices} matrices, got {len(F_list)}"

    strain_values = np.linspace(-delta, delta, num_steps)

    base_strains = [
        (0, 0),  # εxx
        (1, 1),  # εyy
        (2, 2),  # εzz
        (1, 2),  # εyz
        (0, 2),  # εxz
        (0, 1),  # εxy
    ]

    voigt_mapping = {
        (0, 0): 0,
        (1, 1): 1,
        (2, 2): 2,
        (1, 2): 3,
        (2, 1): 3,
        (0, 2): 4,
        (2, 0): 4,
        (0, 1): 5,
        (1, 0): 5,
    }

    for strain_type, (i, j) in enumerate(base_strains):
        for step, strain in enumerate(strain_values):
            matrix_idx = strain_type * num_steps + step
            F = F_list[matrix_idx]

            assert F.shape == (3, 3), f"Invalid shape for matrix {matrix_idx + 1}"
            assert np.allclose(F, F), "Matrix contains NaN or Inf"

            expected_F = np.eye(3)
            if i == j:
                expected_F[i, i] = 1.0 + strain
            else:
                expected_F[i, j] = strain
                expected_F[j, i] = strain

            np.testing.assert_array_almost_equal(
                F,
                expected_F,
                decimal=8,
                err_msg=f"Mismatch at type {strain_type + 1}, step {step + 1}",
            )

            strain_tensor = 0.5 * (F + F.T) - np.eye(3)
            strain_voigt = TensorConverter.to_voigt(strain_tensor, tensor_type="strain")

            expected_strain = np.zeros(6)
            voigt_idx = voigt_mapping[(i, j)]
            if i == j:
                expected_strain[voigt_idx] = strain
            else:
                expected_strain[voigt_idx] = 2 * strain  # 剪切应变乘以 2

            np.testing.assert_array_almost_equal(
                strain_voigt,
                expected_strain,
                decimal=8,
                err_msg=f"Incorrect strain for matrix {matrix_idx + 1}",
            )


def test_apply_deformation(simple_cell):
    """
    测试变形应用，验证:
    1. 晶格矢量的正确更新
    2. 原子位置的正确转换
    3. 周期性边界条件的正确应用
    """
    delta = 0.01
    num_steps = 5
    deformer = Deformer(delta, num_steps=num_steps)
    F_list = deformer.generate_deformation_matrices()

    # 保存初始状态
    original_cell = simple_cell.copy()

    # 测试多个变形情况
    test_indices = [0, num_steps // 2, -1]  # 测试开始、中间和结束的变形

    for test_idx in test_indices:
        # 复制初始晶胞
        test_cell = original_cell.copy()
        F = F_list[test_idx]

        # 应用变形
        deformer.apply_deformation(test_cell, F)

        # 验证晶格矢量
        expected_lattice = F @ original_cell.lattice_vectors
        np.testing.assert_array_almost_equal(
            test_cell.lattice_vectors,
            expected_lattice,
            decimal=8,
            err_msg=f"Incorrect lattice vectors for deformation {test_idx}",
        )

        for i, (orig_atom, deformed_atom) in enumerate(
            zip(original_cell.atoms, test_cell.atoms)
        ):
            # 计算原子的分数坐标
            fractional = np.linalg.solve(
                original_cell.lattice_vectors.T, orig_atom.position
            )

            # 计算期望的原子位置
            expected_pos = expected_lattice.T @ fractional

            # 应用周期性边界条件（如果需要）
            if test_cell.pbc_enabled:
                expected_pos = test_cell.apply_periodic_boundary(expected_pos)

            # 不需要再次对 deformed_atom.position 应用周期性边界条件
            # deformed_atom.position = test_cell.apply_periodic_boundary(deformed_atom.position)

            np.testing.assert_array_almost_equal(
                deformed_atom.position,
                expected_pos,
                decimal=8,
                err_msg=f"Incorrect atomic position {i + 1} for deformation {test_idx + 1}",
            )

        # 验证体积变化
        expected_volume = np.abs(np.linalg.det(F)) * original_cell.volume
        np.testing.assert_almost_equal(
            test_cell.volume,
            expected_volume,
            decimal=8,
            err_msg=f"Incorrect volume for deformation {test_idx + 1}",
        )


def test_strain_symmetry():
    """
    测试应变张量的对称性，验证:
    1. 变形矩阵生成的应变张量是对称的
    2. 应变分量的符号正确性
    """
    delta = 0.01
    num_steps = 3
    deformer = Deformer(delta, num_steps=num_steps)
    F_list = deformer.generate_deformation_matrices()

    for i, F in enumerate(F_list):
        # 计算应变张量
        strain_tensor = 0.5 * (F + F.T) - np.eye(3)

        # 验证对称性
        np.testing.assert_array_almost_equal(
            strain_tensor,
            strain_tensor.T,
            decimal=8,
            err_msg=f"Strain tensor {i} is not symmetric",
        )

        # 验证应变分量
        strain_voigt = TensorConverter.to_voigt(strain_tensor, tensor_type="strain")
        assert len(strain_voigt) == 6, "Incorrect Voigt notation length"
