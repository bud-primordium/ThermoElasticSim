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

        # 六个独立的应变分量
        strain_components = [
            np.array([[delta, 0, 0], [0, 0, 0], [0, 0, 0]]),  # ε_xx
            np.array([[0, 0, 0], [0, delta, 0], [0, 0, 0]]),  # ε_yy
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, delta]]),  # ε_zz
            np.array([[0, delta / 2, 0], [delta / 2, 0, 0], [0, 0, 0]]),  # ε_xy
            np.array([[0, 0, delta / 2], [0, 0, 0], [delta / 2, 0, 0]]),  # ε_xz
            np.array([[0, 0, 0], [0, 0, delta / 2], [0, delta / 2, 0]]),  # ε_yz
        ]

        for epsilon in strain_components:
            F = np.identity(3) + epsilon
            F_list.append(F)

        return F_list

    def apply_deformation(self, cell, deformation_matrix):
        """
        @brief 对晶胞施加变形矩阵。

        @param cell Cell 实例
        @param deformation_matrix 变形矩阵
        """
        cell.apply_deformation(deformation_matrix)
