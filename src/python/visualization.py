# 文件名: visualization.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现晶体结构和应力-应变关系的可视化工具。

"""
可视化模块

包含 Visualizer 类，用于可视化晶胞结构和应力-应变关系
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from .structure import Cell
import logging
import os

logger = logging.getLogger(__name__)

plt.ioff()  # 关闭交互模式，避免 GUI 启动警告


class Visualizer:
    def __init__(self):
        """
        初始化可视化工具类
        """
        pass

    def _ensure_directory_exists(self, filepath):
        """确保文件目录存在"""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

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

    def plot_stress_strain_multiple(
        self, strain_data: np.ndarray, stress_data: np.ndarray, show=True
    ):
        """
        绘制多个应力-应变关系图，每个应力分量对应一个子图

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
        axes : numpy.ndarray
            子图对象数组
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # 使用颜色映射
        cmap = plt.get_cmap("tab10")

        for i in range(6):
            ax = axes[i]
            ax.scatter(
                strain_data[:, i],
                stress_data[:, i],
                color=cmap(i),
                s=50,
                label=f"Component {i+1}",
            )
            ax.set_xlabel(f"Strain {i+1}")
            ax.set_ylabel(f"Stress {i+1} (GPa)")
            ax.set_title(f"Stress {i+1} vs Strain {i+1}")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        if show:
            plt.show()

        return fig, axes

    def create_optimization_animation(
        self, trajectory, filename, title="Optimization", pbc=True, show=True
    ):
        """
        创建优化过程的动画

        Parameters
        ----------
        trajectory : list of dict
            记录的轨迹数据，每个元素包含 'positions', 'volume', 'lattice_vectors'
        filename : str
            保存动画的文件名（包含路径）
        title : str, optional
            动画标题，默认为 "Optimization"
        pbc : bool, optional
            是否应用周期性边界条件，默认为 True
        show : bool, optional
            是否立即显示动画，默认为 True
        """
        self._ensure_directory_exists(filename)
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

        # 同时创建一个日志文件
        log_filename = os.path.splitext(filename)[0] + "_trajectory.log"
        with open(log_filename, "w") as f:
            for i, frame in enumerate(trajectory):
                f.write(f"Frame {i}:\n")
                f.write(f"Positions:\n{frame['positions']}\n")
                f.write(f"Volume: {frame['volume']:.3f}\n")
                f.write(f"Lattice vectors:\n{frame['lattice_vectors']}\n")
                f.write("-" * 50 + "\n")

        def init():
            scatter._offsets3d = ([], [], [])
            return scatter, *quivers, text_volume, text_lattice, text_atoms

        def update(frame):
            data = trajectory[frame]
            positions = np.array(data["positions"])
            volume = data["volume"]
            lattice_vectors = np.array(data["lattice_vectors"])
            num_atoms = positions.shape[0]

            # logger.debug(f"Frame {frame}: Number of atoms = {num_atoms}")
            # logger.debug(f"Positions:\n{positions}")

            scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

            # 更新晶格矢量箭头
            for i, vec in enumerate(lattice_vectors.T):
                quivers[i].remove()
                quivers[i] = ax.quiver(
                    0, 0, 0, vec[0], vec[1], vec[2], color="r", arrow_length_ratio=0.1
                )

            # 更新文本信息，使用透明背景框减少重叠
            text_volume.set_text(f"Volume: {volume:.2f} Å³")
            text_volume.set_position((0.05, 1.08))  # 将位置向上移出绘图区域
            text_volume.set_bbox(dict(facecolor="white", alpha=0.6, edgecolor="none"))

            lattice_str = "\n".join(
                [
                    f"v{i+1}: [{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}]"
                    for i, vec in enumerate(lattice_vectors.T)
                ]
            )
            text_lattice.set_text(f"Lattice Vectors:\n{lattice_str}")
            text_lattice.set_position((0.05, 1.02))  # 调整 lattice 矢量文本框位置
            text_lattice.set_bbox(dict(facecolor="white", alpha=0.6, edgecolor="none"))

            text_atoms.set_text(f"Number of Atoms: {num_atoms}")
            text_atoms.set_position((0.05, 0.96))  # 调整原子数信息的位置
            text_atoms.set_bbox(dict(facecolor="white", alpha=0.6, edgecolor="none"))

            # 保持标题设置不变
            ax.set_title(f"{title} - Step {frame + 1}/{len(trajectory)}")

            # 更新坐标轴范围
            all_positions = positions
            max_range = np.ptp(all_positions, axis=0).max()
            mid_points = np.mean(all_positions, axis=0)

            ax.set_xlim(mid_points[0] - max_range / 2, mid_points[0] + max_range / 2)
            ax.set_ylim(mid_points[1] - max_range / 2, mid_points[1] + max_range / 2)
            ax.set_zlim(mid_points[2] - max_range / 2, mid_points[2] + max_range / 2)

            return scatter, *quivers, text_volume, text_lattice, text_atoms

        ani = FuncAnimation(
            fig,
            update,
            frames=len(trajectory),
            init_func=init,
            blit=False,
            interval=200,
        )

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
