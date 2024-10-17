# src/python/visualization.py

"""
@file visualization.py
@brief 可视化晶胞结构和模拟结果的模块。
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from .structure import Cell


class Visualizer:
    """
    @class Visualizer
    @brief 可视化晶胞结构和模拟结果的类。
    """

    def __init__(self):
        pass

    def plot_cell_structure(self, cell_structure: Cell, show=True):
        """
        @brief 绘制晶体结构的 3D 图形。

        @param cell_structure Cell 实例。
        @param show 是否显示图形，默认为 True。
        @return 返回绘图对象 fig 和 ax。
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for atom in cell_structure.atoms:
            ax.scatter(*atom.position, label=atom.symbol)
        # 绘制晶格矢量
        origin = [0, 0, 0]
        lattice_vectors = cell_structure.lattice_vectors
        for i in range(3):
            vec = lattice_vectors[:, i]
            ax.quiver(*origin, *vec, color="r", arrow_length_ratio=0.1)
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        plt.title("Crystal Structure")
        plt.legend()
        if show:
            plt.show()
        return fig, ax

    def plot_stress_strain(
        self, strain_data: np.ndarray, stress_data: np.ndarray, show=True
    ):
        """
        @brief 绘制应力-应变关系图。

        @param strain_data numpy.ndarray: 应变数据，形状为 (N, 6)。
        @param stress_data numpy.ndarray: 应力数据，形状为 (N, 6)。
        @param show 是否显示图形，默认为 True。
        @return 返回绘图对象 fig 和 ax。
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(6):
            ax.plot(strain_data[:, i], stress_data[:, i], label=f"Stress {i+1}")
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (eV/Å³)")
        ax.set_title("Stress-Strain Relationship")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax
