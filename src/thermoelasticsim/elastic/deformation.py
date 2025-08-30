# 文件名: deformation.py
# 作者: Gilbert Young
# 修改日期: 2025-03-27
# 文件描述: 实现对晶胞施加微小应变的 Deformer 类。

r"""
变形模块

用于生成并对晶胞施加小形变矩阵的工具。

理论基础
--------
小形变近似下，形变梯度可写为：

.. math::
    \mathbf{F} = \mathbf{I} + \boldsymbol{\varepsilon}

其中 :math:`\boldsymbol{\varepsilon}` 为对称的小应变张量。为便于逐分量扫描，
本模块构造 6 个基方向（Voigt 记号）：:math:`\varepsilon_{11}, \varepsilon_{22}, \varepsilon_{33}, \varepsilon_{23}, \varepsilon_{13}, \varepsilon_{12}`。
"""

import logging

import numpy as np

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

    def generate_deformation_matrices(self) -> list[np.ndarray]:
        r"""生成形变矩阵列表

        为 6 个 Voigt 分量依次生成均匀分布的形变序列，采用：

        .. math::
            \mathbf{F}(\delta) = \mathbf{I} + \delta\,\mathbf{E}_k

        其中 :math:`\mathbf{E}_k` 为第 :math:`k` 个分量的基矩阵（剪切为对称填充）。

        Returns
        -------
        list of numpy.ndarray
            形变矩阵列表，每个为 (3, 3)

        Notes
        -----
        预分配结果列表以减少内存分配开销。
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
        r"""对晶胞施加形变矩阵 :math:`\mathbf{F}`

        Parameters
        ----------
        cell : Cell
            包含原子的晶胞对象
        deformation_matrix : numpy.ndarray
            3×3 形变矩阵

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
            # 委托给 Cell 方法，保持坐标与晶格约定一致
            cell.apply_deformation(deformation_matrix)
            logging.info(f"成功应用变形矩阵: {deformation_matrix}")
        except Exception as e:
            logging.error(f"应用变形矩阵失败: {str(e)}")
            raise
