# 文件名: deformation.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现对晶胞施加微小应变的 Deformer 类。

"""
变形模块

包含 Deformer 类，用于生成并对晶胞施加微小应变的变形矩阵
"""

import numpy as np


class Deformer:
    def __init__(self, delta, num_steps=5):
        self.delta = delta
        self.num_steps = num_steps  # 每个应变分量的步数

    def generate_deformation_matrices(self):
        delta_values = np.linspace(-self.delta, self.delta, self.num_steps)
        F_list = []

        # 六个独立的应变分量
        strain_components = [
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),  # ε_xx
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # ε_yy
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),  # ε_zz
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),  # ε_xy
            np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),  # ε_xz
            np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),  # ε_yz
        ]

        for epsilon in strain_components:
            for delta in delta_values:
                strain = delta * epsilon
                F = np.identity(3) + strain
                F_list.append(F)

        return F_list

    def apply_deformation(self, cell, deformation_matrix):
        """
        对晶胞施加变形矩阵

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        deformation_matrix : numpy.ndarray
            变形矩阵
        """
        cell.apply_deformation(deformation_matrix)
