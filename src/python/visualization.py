# 文件名: visualization.py
# 作者: Gilbert Young
# 修改日期: 2024-10-19
# 文件描述: 实现晶体结构和应力-应变关系的可视化工具。

"""
可视化模块

包含 Visualizer 类，用于可视化晶胞结构和应力-应变关系
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from .structure import Cell


class Visualizer:
    """
    可视化类，用于绘制晶体结构和应力-应变关系
    """

    def __init__(self):
        """
        初始化可视化工具类
        """
        pass

    def plot_cell_structure(self, cell_structure: Cell, show=True):
        """
        绘制晶体结构的 3D 图形

        Parameters
        ----------
        cell_structure : Cell
            包含原子的 Cell 实例
        show : bool, optional
            是否立即显示图形，默认为 True

        Returns
        -------
        fig : matplotlib.figure.Figure
            绘制的图形对象
        ax : matplotlib.axes._subplots.Axes3DSubplot
            3D 子图对象
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # 绘制原子位置
        for atom in cell_structure.atoms:
            ax.scatter(*atom.position, label=atom.symbol)  # 按符号标记不同原子

        # 绘制晶格矢量
        origin = [0, 0, 0]
        lattice_vectors = cell_structure.lattice_vectors
        for i in range(3):
            vec = lattice_vectors[:, i]
            ax.quiver(
                *origin, *vec, color="r", arrow_length_ratio=0.1
            )  # 使用箭头表示晶格矢量

        # 设置坐标轴标签
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")

        plt.title("Crystal Structure")
        plt.legend()

        # 显示或返回图形对象
        if show:
            plt.show()

        return fig, ax

    def plot_stress_strain(
        self, strain_data: np.ndarray, stress_data: np.ndarray, show=True
    ):
        """
        绘制应力-应变关系图

        Parameters
        ----------
        strain_data : numpy.ndarray
            应变数据，形状为 (N, 6)
        stress_data : numpy.ndarray
            应力数据，形状为 (N, 6)
        show : bool, optional
            是否立即显示图形，默认为 True

        Returns
        -------
        fig : matplotlib.figure.Figure
            绘制的图形对象
        ax : matplotlib.axes._subplots.AxesSubplot
            2D 子图对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # 循环绘制每个应力分量的应力-应变关系
        for i in range(6):
            ax.plot(strain_data[:, i], stress_data[:, i], label=f"Stress {i+1}")

        # 设置坐标轴标签和图形标题
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (eV/Å³)")
        ax.set_title("Stress-Strain Relationship")

        ax.legend()  # 显示图例
        ax.grid(True)  # 显示网格

        plt.tight_layout()

        # 显示或返回图形对象
        if show:
            plt.show()

        return fig, ax
