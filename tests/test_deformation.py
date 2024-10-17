# tests/test_deformation.py

import unittest
import numpy as np
import sys
import os

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from python.deformation import Deformer


class TestDeformation(unittest.TestCase):
    def test_generate_deformation_matrices(self):
        delta = 0.01  # 1% 变形
        deformer = Deformer(delta)
        deformation_matrices = deformer.generate_deformation_matrices()

        # 预期生成 6 个变形矩阵（正交基的变形）
        self.assertEqual(len(deformation_matrices), 6)

        # 检查每个变形矩阵的正确性
        for i, F in enumerate(deformation_matrices):
            # 检查是否为 3x3 矩阵
            self.assertEqual(F.shape, (3, 3))

            # 根据变形类型，检查特定元素是否有变形
            if i < 3:
                # 轴向拉伸/压缩
                self.assertAlmostEqual(F[i, i], 1 + delta)
                for j in range(3):
                    if j != i:
                        self.assertAlmostEqual(F[i, j], 0.0)
            else:
                # 剪切变形
                shear_index = i - 3
                self.assertAlmostEqual(F[0, 1], delta if shear_index == 0 else 0.0)
                self.assertAlmostEqual(F[0, 2], delta if shear_index == 1 else 0.0)
                self.assertAlmostEqual(F[1, 2], delta if shear_index == 2 else 0.0)
                # 对角线元素应为1
                for j in range(3):
                    self.assertAlmostEqual(F[j, j], 1.0)


if __name__ == "__main__":
    unittest.main()
