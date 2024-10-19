# 文件名: deformation.py
# 作者: Gilbert Young
# 修改日期: 2024-10-19
# 文件描述: 实现对晶胞施加微小应变的 Deformer 类。

"""
变形模块

包含 Deformer 类，用于生成并对晶胞施加微小应变的变形矩阵
"""

import numpy as np


class Deformer:
    """
    施加微小应变以生成变形矩阵的类

    Parameters
    ----------
    delta : float
        微小应变量
    """

    def __init__(self, delta):
        """初始化 Deformer 对象"""
        self.delta = delta

    def generate_deformation_matrices(self):
        """
        生成用于施加应变的变形矩阵列表

        Returns
        -------
        list of numpy.ndarray
            变形矩阵列表
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
        对晶胞施加变形矩阵

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        deformation_matrix : numpy.ndarray
            变形矩阵
        """
        cell.apply_deformation(deformation_matrix)
