# tests/test_deformation.py

import unittest
import numpy as np
from python.deformation import Deformer


class TestDeformation(unittest.TestCase):
    """
    @class TestDeformation
    @brief 测试 Deformer 类的功能
    """

    def test_generate_deformation_matrices(self):
        """
        @brief 测试生成变形矩阵的方法
        """
        delta = 0.01  # 1% 的应变
        deformer = Deformer(delta)
        deformation_matrices = deformer.generate_deformation_matrices()

        # 检查是否生成了 6 个变形矩阵
        self.assertEqual(len(deformation_matrices), 6)

        # 检查每个变形矩阵的正确性
        for i, F in enumerate(deformation_matrices):
            # 检查矩阵是否为 3x3
            self.assertEqual(F.shape, (3, 3))

            # 检查矩阵是否为单位矩阵加上微小变形
            expected_F = np.identity(3)
            if i < 3:
                expected_F[i, i] += delta
            else:
                shear_indices = [(0, 1), (0, 2), (1, 2)]
                idx = i - 3
                i_shear, j_shear = shear_indices[idx]
                expected_F[i_shear, j_shear] += delta
            np.testing.assert_array_almost_equal(F, expected_F)


if __name__ == "__main__":
    unittest.main()
