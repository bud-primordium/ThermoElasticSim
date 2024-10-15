# src/python/visualization.py

"""
@file visualization.py
@brief 可视化晶胞结构和模拟结果的模块。
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List

from src.python.structure import Cell


class Visualizer:
    """
    @class Visualizer
    @brief 可视化晶胞结构和模拟结果的类。
    """

    def __init__(self) -> None:
        """
        @brief 初始化 Visualizer 实例。
        """
        pass

    def plot_cell_structure(self, cell_structure: Cell) -> None:
        """
        @brief 绘制晶体结构的3D图形。

        @param cell_structure Cell 实例。
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for atom in cell_structure.atoms:
            ax.scatter(*atom.position, label=atom.symbol)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Crystal Structure")
        plt.legend()
        plt.show()

    def plot_stress_strain(
        self, strain_data: np.ndarray, stress_data: np.ndarray
    ) -> None:
        """
        @brief 绘制应力-应变关系图。

        @param strain_data numpy.ndarray: 应变数据，形状为 (N, 6)。
        @param stress_data numpy.ndarray: 应力数据，形状为 (N, 6)。
        """
        plt.figure(figsize=(10, 6))
        for i in range(6):
            plt.plot(strain_data[:, i], stress_data[:, i], label=f"Stress {i+1}")
        plt.xlabel("Strain")
        plt.ylabel("Stress")
        plt.title("Stress-Strain Relationship")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
