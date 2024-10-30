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
import os
from .structure import Cell
from matplotlib.animation import FuncAnimation, PillowWriter


class Visualizer:
    def __init__(self, save_path="./output"):
        """
        初始化可视化工具类

        Parameters
        ----------
        save_path : str
            图片和动画的保存路径，默认为 './output'
        """
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)  # 确保路径存在

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

        # 设置相同的轴比例
        self.set_axes_equal(ax)

        # 显示或返回图形对象
        if show:
            plt.show()

        return fig, ax

    def set_axes_equal(self, ax):
        """
        使3D坐标轴比例相同
        """
        limits = np.array(
            [
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ]
        )
        origin = np.min(limits, axis=1)
        size = np.max(limits, axis=1) - origin
        max_size = max(size)
        origin -= (max_size - size) / 2
        ax.set_xlim3d(origin[0], origin[0] + max_size)
        ax.set_ylim3d(origin[1], origin[1] + max_size)
        ax.set_zlim3d(origin[2], origin[2] + max_size)

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

    def create_optimization_animation(
        self, trajectory, filename, title="Optimization", pbc=True, show=True
    ):
        filename = os.path.join(self.save_path, filename)  # 使用保存路径
        """
        创建优化过程的动画

        Parameters
        ----------
        trajectory : list of dict
            记录的轨迹数据，每个元素包含 'positions', 'volume', 'lattice_vectors'
        filename : str
            保存动画的文件名（支持 .gif, .mp4 等格式）
        title : str, optional
            动画标题，默认为 "Optimization"
        pbc : bool, optional
            是否应用周期性边界条件，默认为 True
        show : bool, optional
            是否立即显示动画，默认为 True
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # 初始化图像
        scatter = ax.scatter([], [], [], color="b", s=20)
        quivers = [
            ax.quiver(0, 0, 0, 0, 0, 0, color="r", arrow_length_ratio=0.1)
            for _ in range(3)
        ]
        text_volume = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
        text_lattice = ax.text2D(0.05, 0.90, "", transform=ax.transAxes)
        text_atoms = ax.text2D(0.05, 0.85, "", transform=ax.transAxes)

        def init():
            scatter._offsets3d = ([], [], [])
            for quiver in quivers:
                quiver.remove()
            quivers[:] = [
                ax.quiver(0, 0, 0, 0, 0, 0, color="r", arrow_length_ratio=0.1)
                for _ in range(3)
            ]
            text_volume.set_text("")
            text_lattice.set_text("")
            text_atoms.set_text("")
            return scatter, *quivers, text_volume, text_lattice, text_atoms

        def update(frame):
            data = trajectory[frame]
            positions = data["positions"]
            volume = data["volume"]
            lattice_vectors = data["lattice_vectors"]
            num_atoms = positions.shape[0]

            scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

            # 更新晶格矢量箭头
            for i, vec in enumerate(lattice_vectors.T):
                quivers[i].remove()
                quivers[i] = ax.quiver(
                    0, 0, 0, vec[0], vec[1], vec[2], color="r", arrow_length_ratio=0.1
                )

            # 更新文本信息
            text_volume.set_text(f"Volume: {volume:.2f} Å³")
            lattice_str = "\n".join(
                [
                    f"v{i+1}: [{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}]"
                    for i, vec in enumerate(lattice_vectors)
                ]
            )
            text_lattice.set_text(f"Lattice Vectors:\n{lattice_str}")
            text_atoms.set_text(f"Number of Atoms: {num_atoms}")

            ax.set_title(f"{title} - Step {frame + 1}/{len(trajectory)}")

            return scatter, *quivers, text_volume, text_lattice, text_atoms

        ani = FuncAnimation(
            fig,
            update,
            frames=len(trajectory),
            init_func=init,
            blit=False,
            interval=200,
        )

        # 设置相同的轴比例
        self.set_axes_equal(ax)

        # 保存动画
        ani.save(filename, writer=PillowWriter(fps=5))
        if show:
            plt.show()
        plt.close(fig)

    def create_stress_strain_animation(
        self, strain_data, stress_data, filename, show=True
    ):
        """
        创建应力-应变关系的动画

        Parameters
        ----------
        strain_data : numpy.ndarray
            应变数据，形状为 (N, 6)
        stress_data : numpy.ndarray
            应力数据，形状为 (N, 6)
        filename : str
            保存动画的文件名（支持 .gif, .mp4 等格式）
        show : bool, optional
            是否立即显示动画，默认为 True
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        lines = [ax.plot([], [], label=f"Stress {i+1}")[0] for i in range(6)]
        ax.set_xlim(strain_data.min(), strain_data.max())
        ax.set_ylim(stress_data.min(), stress_data.max())
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (eV/Å³)")
        ax.set_title("Stress-Strain Relationship Animation")
        ax.legend()
        ax.grid(True)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def update(frame):
            for i, line in enumerate(lines):
                line.set_data(strain_data[:frame, i], stress_data[:frame, i])
            return lines

        ani = FuncAnimation(
            fig,
            update,
            frames=len(strain_data) + 1,
            init_func=init,
            blit=False,
            interval=100,
        )

        plt.tight_layout()
        ani.save(filename, writer=PillowWriter(fps=5))
        if show:
            plt.show()
        plt.close(fig)
