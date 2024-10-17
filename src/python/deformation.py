# src/python/deformation.py

import numpy as np


class Deformer:
    """
    @class Deformer
    @brief 施加微小应变以生成变形矩阵的类。
    """

    def __init__(self, delta):
        """
        @param delta 微小应变量
        """
        self.delta = delta

    def generate_deformation_matrices(self):
        """
        @brief 生成用于施加应变的变形矩阵列表。

        @return 变形矩阵列表
        """
        delta = self.delta
        F_list = []

        # 轴向拉伸/压缩
        for i in range(3):
            F = np.identity(3)
            F[i, i] += delta
            F_list.append(F)

        # 剪切变形
        shear_indices = [(0, 1), (0, 2), (1, 2)]
        for i, j in shear_indices:
            F = np.identity(3)
            F[i, j] += delta
            F_list.append(F)

        return F_list

    def apply_deformation(self, cell, deformation_matrix):
        """
        @brief 对晶胞施加变形矩阵。

        @param cell Cell 实例
        @param deformation_matrix 变形矩阵
        """
        cell.apply_deformation(deformation_matrix)
