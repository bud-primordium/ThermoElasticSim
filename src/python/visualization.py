# 文件名: visualization.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
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

        # 检查原子是否都是同一种元素
        atom_symbols = {atom.symbol for atom in cell_structure.atoms}
        single_element = len(atom_symbols) == 1
        color = "b" if single_element else None  # 设置为蓝色，如果单一元素

        if single_element:
            # 如果只有一种元素，绘制为单一颜色
            positions = [atom.position for atom in cell_structure.atoms]
            ax.scatter(
                *np.array(positions).T, color=color, label=next(iter(atom_symbols))
            )
        else:
            # 不同元素使用不同颜色和标签
            for atom in cell_structure.atoms:
                ax.scatter(*atom.position, label=atom.symbol)

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

        # 只有在多种元素情况下才显示图例
        if not single_element:
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
