# 文件名: deformation.py
# 作者: Gilbert Young
# 修改日期: 2025-03-27
# 文件描述: 实现对晶胞施加微小应变的 Deformer 类。

"""
变形模块

包含 Deformer 类，用于生成并对晶胞施加微小应变的变形矩阵

Classes:
    Deformer: 生成变形矩阵并应用于晶胞的类
"""

import numpy as np
from typing import List, Optional
import logging
from thermoelasticsim.core.structure import Cell


class Deformer:
    """生成变形矩阵并应用于晶胞

    Attributes
    ----------
    delta : float
        应变幅度，默认0.01
    num_steps : int
        每个应变分量的步数，默认10
    logger : logging.Logger
        日志记录器
    """

    def __init__(self, delta: float = 0.01, num_steps: int = 10):
        """初始化Deformer

        Parameters
        ----------
        delta : float, optional
            应变幅度，建议范围0.001-0.05 (默认: 0.01)
        num_steps : int, optional
            每个应变分量的步数，必须大于0 (默认: 10)

        Raises
        ------
        ValueError
            如果delta超出合理范围或num_steps非正
        """
        if not 0.001 <= delta <= 0.05:
            raise ValueError("delta必须在0.001到0.05之间")
        if num_steps <= 0:
            raise ValueError("num_steps必须大于0")

        self.delta = delta
        self.num_steps = num_steps

    # 类常量定义应变分量基
    STRAIN_COMPONENTS = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),  # εxx
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # εyy
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),  # εzz
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),  # εyz
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),  # εxz
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),  # εxy
    ]

    def generate_deformation_matrices(self) -> List[np.ndarray]:
        """生成变形矩阵列表

        生成6个应变分量(εxx,εyy,εzz,εyz,εxz,εxy)的变形矩阵，
        每个分量生成num_steps个变形矩阵

        Returns
        -------
        list
            包含变形矩阵的列表，每个矩阵是3x3 numpy数组

        Notes
        -----
        使用向量化计算优化性能，预分配结果列表空间
        """
        delta_values = np.linspace(-self.delta, self.delta, self.num_steps)
        total_matrices = len(self.STRAIN_COMPONENTS) * len(delta_values)
        F_list = [None] * total_matrices  # 预分配空间

        for i, epsilon in enumerate(self.STRAIN_COMPONENTS):
            for j, delta in enumerate(delta_values):
                strain = delta * epsilon
                F = np.identity(3) + strain
                F_list[i * len(delta_values) + j] = F

        return F_list

    def apply_deformation(self, cell: Cell, deformation_matrix: np.ndarray) -> None:
        """对晶胞施加变形矩阵

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        deformation_matrix : numpy.ndarray
            3x3变形矩阵

        Raises
        ------
        ValueError
            如果变形矩阵不是3x3矩阵或行列式接近零
        """
        if deformation_matrix.shape != (3, 3):
            raise ValueError("变形矩阵必须是3x3矩阵")

        det = np.linalg.det(deformation_matrix)
        if np.isclose(det, 0, atol=1e-10):
            raise ValueError("变形矩阵行列式接近零，可能导致数值不稳定")

        try:
            cell.apply_deformation(deformation_matrix)
            logging.info(f"成功应用变形矩阵: {deformation_matrix}")
        except Exception as e:
            logging.error(f"应用变形矩阵失败: {str(e)}")
            raise
