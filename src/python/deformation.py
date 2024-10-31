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
    def __init__(self, delta=0.01, num_steps=10):
        self.delta = delta
        self.num_steps = num_steps

    def generate_deformation_matrices(self):
        delta_values = np.linspace(-self.delta, self.delta, self.num_steps)
        F_list = []

        # 六个独立的应变分量（不需要在这里处理2倍关系）
        strain_components = [
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),  # εxx
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # εyy
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),  # εzz
            np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),  # εyz
            np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),  # εxz
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),  # εxy
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
