# 文件名: visualization.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现晶体结构和应力-应变关系的可视化工具。

"""
可视化模块

包含 Visualizer 类，用于可视化晶胞结构和应力-应变关系
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.metrics import r2_score

from .structure import Cell

# 配置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

plt.ioff()  # 关闭交互模式，避免 GUI 启动警告


class Visualizer:
    def __init__(self):
        """
        初始化可视化工具类
        """
        pass

    def _ensure_directory_exists(self, filepath):
        """
        确保文件目录存在

        Parameters
        ----------
        filepath : str
            文件路径
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

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
        logger.info("Plotting crystal structure.")
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # 检查原子是否都是同一种元素
        atom_symbols = {atom.symbol for atom in cell_structure.atoms}
        single_element = len(atom_symbols) == 1
        color = "b" if single_element else None  # 设置为蓝色，如果单一元素

        if single_element:
            # 如果只有一种元素，绘制为单一颜色
            positions = np.array([atom.position for atom in cell_structure.atoms])
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=color,
                label=next(iter(atom_symbols)),
                s=50,
            )
            logger.debug(
                f"Plotted atoms with single element: {next(iter(atom_symbols))}"
            )
        else:
            # 不同元素使用不同颜色和标签
            for atom in cell_structure.atoms:
                ax.scatter(
                    atom.position[0],
                    atom.position[1],
                    atom.position[2],
                    label=atom.symbol,
                    s=50,
                )
            logger.debug("Plotted atoms with multiple elements.")

        # 绘制晶格矢量
        origin = np.array([0, 0, 0])
        lattice_vectors = cell_structure.lattice_vectors
        for i in range(3):
            vec = lattice_vectors[:, i]
            ax.quiver(
                *origin, *vec, color="r", arrow_length_ratio=0.1, linewidth=1.5
            )  # 使用箭头表示晶格矢量
            logger.debug(f"Plotted lattice vector v{i+1}: {vec}")

        # 设置坐标轴标签
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")

        plt.title("Crystal Structure")

        # 只有在多种元素情况下才显示图例
        if not single_element:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        # 设置相同的轴比例
        self.set_axes_equal(ax)

        # 显示或返回图形对象
        if show:
            plt.show()

        logger.info("Crystal structure plot completed.")
        return fig, ax

    def set_axes_equal(self, ax):
        """
        使3D坐标轴比例相同

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot
            3D 子图对象
        """
        logger.debug("Setting equal aspect ratio for axes.")
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
        logger.debug(f"Axes limits set to origin: {origin}, size: {max_size}")

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
        logger.info("Plotting multiple stress-strain relationships.")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # 使用颜色映射
        cmap = plt.get_cmap("tab10")

        components = ["11", "22", "33", "23", "13", "12"]

        for i in range(6):
            ax = axes[i]
            ax.scatter(
                strain_data[:, i],
                stress_data[:, i],
                color=cmap(i),
                s=50,
                label=f"Component {components[i]}",
                alpha=0.7,
            )
            ax.set_xlabel(f"Strain {components[i]}")
            ax.set_ylabel(f"Stress {components[i]} (GPa)")
            ax.set_title(f"Stress {components[i]} vs Strain {components[i]}")
            ax.legend()
            ax.grid(True)
            logger.debug(f"Plotted stress-strain for component {components[i]}.")

        plt.tight_layout()

        if show:
            plt.show()

        logger.info("Multiple stress-strain plots completed.")
        return fig, axes

    def plot_deformation_stress_strain(
        self, strains, stresses, mode_index, save_path, show=False
    ):
        """
        为特定变形方式绘制所有应力分量的响应图

        Parameters
        ----------
        strains : numpy.ndarray
            该变形方式下的应变数据
        stresses : numpy.ndarray
            该变形方式下的应力数据
        mode_index : int
            变形模式索引 (0-5)
        save_path : str
            保存路径
        show : bool, optional
            是否显示图像，默认为 False
        """
        logger.info(f"Plotting deformation stress-strain for mode index {mode_index}.")
        components = ["11", "22", "33", "23", "13", "12"]
        mode_labels = {
            "11": "ε₁₁",
            "22": "ε₂₂",
            "33": "ε₃₃",
            "23": "ε₂₃",
            "13": "ε₁₃",
            "12": "ε₁₂",
        }
        stress_labels = {
            "11": "σ₁₁",
            "22": "σ₂₂",
            "33": "σ₃₃",
            "23": "σ₂₃",
            "13": "σ₁₃",
            "12": "σ₁₂",
        }

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        current_mode = components[mode_index]
        fig.suptitle(
            f"Deformation Mode: {mode_labels[current_mode]}", fontsize=14, y=1.02
        )

        # 对每个应力分量绘制其对该变形的响应
        for i, component in enumerate(components):
            ax = axes[i]

            # 绘制散点
            ax.scatter(
                strains[:, mode_index],
                stresses[:, i],
                color="blue",
                alpha=0.6,
                label="Data",
                s=30,
            )

            # 添加拟合线
            if len(strains[:, mode_index]) > 1:
                coeffs = np.polyfit(strains[:, mode_index], stresses[:, i], 1)
                fit_line = np.poly1d(coeffs)
                strain_range = np.linspace(
                    strains[:, mode_index].min(),
                    strains[:, mode_index].max(),
                    100,
                )
                ax.plot(
                    strain_range, fit_line(strain_range), "r-", alpha=0.8, label="Fit"
                )

                # 计算并显示R^2
                y_pred = fit_line(strains[:, mode_index])
                r2 = r2_score(stresses[:, i], y_pred)
                logger.debug(
                    f"Mode {mode_index}, Component {component}: R^2 = {r2:.4f}"
                )
            else:
                r2 = 1.0  # 单点时R^2默认1

            # 设置标签和标题
            ax.set_xlabel(f"{mode_labels[current_mode]} (Strain)")
            ax.set_ylabel(f"{stress_labels[component]} (GPa)")
            ax.set_title(
                f"{stress_labels[component]} vs {mode_labels[current_mode]}\nR^2 = {r2:.4f}"
            )
            ax.grid(True, alpha=0.3)
            if i == 0:  # 只在第一个子图显示图例
                ax.legend()

            logger.debug(
                f"Plotted deformation stress-strain for component {component}."
            )

        plt.tight_layout()

        # 保存图片
        filename = os.path.join(save_path, f"deformation_{components[mode_index]}.png")
        self._ensure_directory_exists(filename)
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved deformation stress-strain plot to {filename}")

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
        logger.info(f"Creating optimization animation: {filename}")
        self._ensure_directory_exists(filename)
        fig = plt.figure(figsize=(8, 6))
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
        logger.debug(f"Saved trajectory log to {log_filename}")

        def init():
            scatter._offsets3d = ([], [], [])
            for quiver in quivers:
                quiver.remove()
            quivers.clear()
            for _ in range(3):
                quivers.append(
                    ax.quiver(0, 0, 0, 0, 0, 0, color="r", arrow_length_ratio=0.1)
                )
            text_volume.set_text("")
            text_lattice.set_text("")
            text_atoms.set_text("")
            return scatter, *quivers, text_volume, text_lattice, text_atoms

        def update(frame):
            data = trajectory[frame]
            positions = np.array(data["positions"])
            volume = data["volume"]
            lattice_vectors = np.array(data["lattice_vectors"])
            num_atoms = positions.shape[0]

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
        logger.info(f"Saved optimization animation to {filename}")

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
        logger.info(f"Creating stress-strain animation: {filename}")
        self._ensure_directory_exists(filename)
        fig, ax = plt.subplots(figsize=(10, 6))

        components = ["11", "22", "33", "23", "13", "12"]
        lines = [ax.plot([], [], label=f"Stress {comp}")[0] for comp in components]
        ax.set_xlim(strain_data.min() * 1.05, strain_data.max() * 1.05)
        ax.set_ylim(stress_data.min() * 1.05, stress_data.max() * 1.05)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (GPa)")
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
            ax.set_title(
                f"Stress-Strain Relationship Animation - Frame {frame}/{len(strain_data)}"
            )
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
        logger.info(f"Saved stress-strain relationship animation to {filename}")

        if show:
            plt.show()
        plt.close(fig)

    def create_simulation_animation(self, trajectory_data, filename, interval=100):
        """
        创建分子动力学模拟过程的动画

        Parameters
        ----------
        trajectory_data : list of dict
            轨迹数据列表，每个字典包含:
            - positions: array_like, 原子位置
            - cell: Cell object, 晶胞对象
            - time: float, 时间
            - temperature: float, 温度
            - energy: float, 能量
        filename : str
            保存动画的文件名
        interval : int, optional
            帧之间的时间间隔(毫秒)，默认100
        """
        fig = plt.figure(figsize=(15, 8))

        # 创建子图
        ax_structure = fig.add_subplot(121, projection="3d")
        ax_props = fig.add_subplot(122)

        # 初始化原子散点图
        scatter = ax_structure.scatter([], [], [], c="b", marker="o")

        # 初始化晶格箭头
        quivers = [ax_structure.quiver(0, 0, 0, 0, 0, 0, color="r") for _ in range(3)]

        # 初始化属性图
        (temp_line,) = ax_props.plot([], [], "r-", label="Temperature (K)")
        (energy_line,) = ax_props.plot([], [], "b-", label="Energy (eV)")

        # 存储时间数据
        times = [frame["time"] for frame in trajectory_data]
        temperatures = [frame["temperature"] for frame in trajectory_data]
        energies = [frame["energy"] for frame in trajectory_data]

        def init():
            scatter._offsets3d = ([], [], [])
            temp_line.set_data([], [])
            energy_line.set_data([], [])
            return scatter, temp_line, energy_line

        def update(frame):
            # 更新结构视图
            positions = trajectory_data[frame]["positions"]
            cell = trajectory_data[frame]["cell"]

            # 更新原子位置
            scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

            # 更新晶格向量
            for i, quiver in enumerate(quivers):
                quiver.remove()
                vec = cell.lattice_vectors[:, i]
                quivers[i] = ax_structure.quiver(
                    0, 0, 0, vec[0], vec[1], vec[2], color="r"
                )

            # 更新属性图
            temp_line.set_data(times[:frame], temperatures[:frame])
            energy_line.set_data(times[:frame], energies[:frame])

            return scatter, temp_line, energy_line, *quivers

        # 设置坐标轴标签
        ax_structure.set_xlabel("X (Å)")
        ax_structure.set_ylabel("Y (Å)")
        ax_structure.set_zlabel("Z (Å)")

        ax_props.set_xlabel("Time (ps)")
        ax_props.set_ylabel("Value")
        ax_props.legend()
        ax_props.grid(True)

        # 创建动画
        anim = FuncAnimation(
            fig,
            update,
            frames=len(trajectory_data),
            init_func=init,
            blit=True,
            interval=interval,
        )

        # 保存动画
        anim.save(filename, writer="pillow")
        plt.close()
