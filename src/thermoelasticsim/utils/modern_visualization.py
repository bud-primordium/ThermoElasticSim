#!/usr/bin/env python3
"""
现代化可视化系统

基于plotly的交互式可视化，支持：
- 3D结构可视化
- 轨迹动画
- 应力应变关系
- 能量演化
- 交互式数据探索

Author: Gilbert Young
Created: 2025-08-15
"""

import logging
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from plotly.subplots import make_subplots

from thermoelasticsim.core.structure import Cell
from thermoelasticsim.utils.plot_config import plt
from thermoelasticsim.utils.trajectory import TrajectoryReader

logger = logging.getLogger(__name__)


class ModernVisualizer:
    """
    现代化可视化工具

    提供交互式3D可视化、动画生成、数据分析图表等功能。

    Examples
    --------
    >>> vis = ModernVisualizer()
    >>> vis.plot_structure_3d(cell)
    >>> vis.create_trajectory_animation('trajectory.h5', 'animation.mp4')
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        初始化可视化器

        Parameters
        ----------
        theme : str
            plotly主题
        """
        self.theme = theme
        self.default_colors = px.colors.qualitative.Plotly

    def plot_structure_3d(
        self,
        cell: Cell,
        show_box: bool = True,
        show_bonds: bool = False,
        bond_cutoff: float = 3.0,
        title: str = "Crystal Structure",
        save_html: str | None = None,
    ) -> go.Figure:
        """
        创建交互式3D结构图

        Parameters
        ----------
        cell : Cell
            晶胞对象
        show_box : bool
            是否显示晶格盒子
        show_bonds : bool
            是否显示键
        bond_cutoff : float
            成键截断距离
        title : str
            图标题
        save_html : str, optional
            保存为HTML文件

        Returns
        -------
        fig : plotly.graph_objects.Figure
            交互式3D图
        """
        positions = cell.get_positions()

        # 原子散点图
        atom_trace = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(
                size=8, color="blue", opacity=0.8, line=dict(width=0.5, color="black")
            ),
            text=[f"Atom {i}: {atom.symbol}" for i, atom in enumerate(cell.atoms)],
            hovertemplate="%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}",
            name="Atoms",
        )

        traces = [atom_trace]

        # 晶格盒子
        if show_box:
            box_traces = self._create_box_traces(cell.lattice_vectors)
            traces.extend(box_traces)

        # 原子键
        if show_bonds:
            bond_traces = self._create_bond_traces(cell, bond_cutoff)
            traces.extend(bond_traces)

        # 创建图形
        fig = go.Figure(data=traces)

        # 设置布局
        fig.update_layout(
            title=title,
            template=self.theme,
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            showlegend=True,
            width=800,
            height=600,
        )

        if save_html:
            fig.write_html(save_html)
            logger.info(f"保存3D结构图到: {save_html}")

        return fig

    def _create_box_traces(self, lattice_vectors: np.ndarray) -> list[go.Scatter3d]:
        """创建晶格盒子的线条"""
        traces = []

        # 8个顶点
        vertices = np.array(
            [
                [0, 0, 0],
                lattice_vectors[0],
                lattice_vectors[1],
                lattice_vectors[2],
                lattice_vectors[0] + lattice_vectors[1],
                lattice_vectors[0] + lattice_vectors[2],
                lattice_vectors[1] + lattice_vectors[2],
                lattice_vectors[0] + lattice_vectors[1] + lattice_vectors[2],
            ]
        )

        # 12条边
        edges = [
            (0, 1),
            (0, 2),
            (0, 3),  # 从原点出发
            (1, 4),
            (1, 5),  # 从v1出发
            (2, 4),
            (2, 6),  # 从v2出发
            (3, 5),
            (3, 6),  # 从v3出发
            (4, 7),
            (5, 7),
            (6, 7),  # 到最远顶点
        ]

        for i, (start, end) in enumerate(edges):
            trace = go.Scatter3d(
                x=[vertices[start, 0], vertices[end, 0]],
                y=[vertices[start, 1], vertices[end, 1]],
                z=[vertices[start, 2], vertices[end, 2]],
                mode="lines",
                line=dict(color="red", width=2),
                showlegend=i == 0,
                name="Box",
                legendgroup="box",
            )
            traces.append(trace)

        return traces

    def _create_bond_traces(self, cell: Cell, cutoff: float) -> list[go.Scatter3d]:
        """创建原子键的线条"""
        traces = []
        positions = cell.get_positions()

        for i in range(len(cell.atoms)):
            for j in range(i + 1, len(cell.atoms)):
                # 计算距离（考虑PBC）
                rij = positions[j] - positions[i]
                if cell.pbc_enabled:
                    rij = cell.minimum_image(rij)

                dist = np.linalg.norm(rij)

                if dist < cutoff:
                    trace = go.Scatter3d(
                        x=[positions[i, 0], positions[j, 0]],
                        y=[positions[i, 1], positions[j, 1]],
                        z=[positions[i, 2], positions[j, 2]],
                        mode="lines",
                        line=dict(color="gray", width=1),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                    traces.append(trace)

        return traces

    def plot_trajectory_frame(
        self, trajectory_file: str, frame_idx: int = 0, **kwargs
    ) -> go.Figure:
        """
        可视化轨迹中的单帧

        Parameters
        ----------
        trajectory_file : str
            轨迹文件
        frame_idx : int
            帧索引
        **kwargs
            传递给plot_structure_3d的参数

        Returns
        -------
        fig : plotly.graph_objects.Figure
            3D结构图
        """
        reader = TrajectoryReader(trajectory_file)
        reader.open()

        frame = reader.read_frame(frame_idx)

        # 创建临时Cell对象
        # 这里需要从轨迹中重建Cell
        # 简化处理，只显示位置

        positions = frame["positions"]
        box = frame.get("box", np.eye(3) * 10)

        # 创建散点图
        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker=dict(size=6, color="blue"),
                name=f"Frame {frame_idx}",
            )
        )

        # 添加盒子
        if "box" in frame:
            box_traces = self._create_box_traces(box)
            for trace in box_traces:
                fig.add_trace(trace)

        fig.update_layout(
            title=f"Trajectory Frame {frame_idx}",
            template=self.theme,
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                aspectmode="cube",
            ),
        )

        reader.close()

        return fig

    def create_trajectory_animation_plotly(
        self, trajectory_file: str, output_html: str, skip: int = 1, duration: int = 100
    ):
        """
        创建交互式轨迹动画（HTML）

        Parameters
        ----------
        trajectory_file : str
            轨迹文件
        output_html : str
            输出HTML文件
        skip : int
            跳帧数
        duration : int
            每帧持续时间（毫秒）
        """
        reader = TrajectoryReader(trajectory_file)
        reader.open()

        info = reader.get_trajectory_info()
        n_frames = info["n_frames"]

        # 读取所有帧（可能需要采样）
        frames_data = []
        frame_indices = list(range(0, n_frames, skip))

        for idx in frame_indices:
            frame = reader.read_frame(idx)
            frames_data.append(frame)

        reader.close()

        # 创建动画帧
        animation_frames = []

        for i, frame in enumerate(frames_data):
            positions = frame["positions"]

            # 获取增强的物理信息
            step_type = frame.get("step_type", "unknown")
            strain_value = frame.get("strain_value", 0.0)
            energy = frame.get("energy", 0.0)
            lattice_a = frame.get("lattice_a", 0.0)
            lattice_b = frame.get("lattice_b", 0.0)
            lattice_c = frame.get("lattice_c", 0.0)
            lattice_alpha = frame.get("lattice_alpha", 90.0)
            lattice_beta = frame.get("lattice_beta", 90.0)
            lattice_gamma = frame.get("lattice_gamma", 90.0)
            volume = frame.get("volume", 0.0)
            converged = frame.get("converged", True)
            description = frame.get("description", f"{step_type} | 帧{i}")

            # 根据步骤类型确定颜色
            if step_type == "base_state":
                color = "green"
                symbol = "circle"
            elif step_type == "before_internal_relax":
                color = "red"
                symbol = "square"
            elif step_type == "after_internal_relax":
                color = "blue"
                symbol = "diamond"
            else:
                color = "gray"
                symbol = "circle"

            # 创建详细的悬停信息 - 修复所有编码和对应问题
            converged_text = "converged" if converged else "failed"

            # 确保step_type是字符串而不是字节对象
            if isinstance(step_type, bytes):
                step_type = step_type.decode("utf-8")
            step_type = str(step_type)

            # 判断是否为三斜晶胞
            is_triclinic = abs(lattice_gamma - 90.0) > 0.1

            if step_type == "base_state":
                hover_text = (
                    f"Base State<br>"
                    f"Energy: {energy:.3f} eV<br>"
                    f"Lattice: {lattice_a:.3f} A<br>"
                    f"Volume: {volume:.1f} A3<br>"
                    f"Type: cubic"
                )
            else:
                if is_triclinic:
                    hover_text = (
                        f"{step_type}<br>"
                        f"Strain: {strain_value:.4f}<br>"
                        f"Energy: {energy:.3f} eV<br>"
                        f"a={lattice_a:.3f} b={lattice_b:.3f} c={lattice_c:.3f} A<br>"
                        f"alpha={lattice_alpha:.1f} beta={lattice_beta:.1f} gamma={lattice_gamma:.1f} deg<br>"
                        f"Volume: {volume:.1f} A3<br>"
                        f"Status: {converged_text}"
                    )
                else:
                    hover_text = (
                        f"{step_type}<br>"
                        f"Strain: {strain_value:.4f}<br>"
                        f"Energy: {energy:.3f} eV<br>"
                        f"Lattice: {lattice_a:.3f} A<br>"
                        f"Volume: {volume:.1f} A3<br>"
                        f"Status: {converged_text}"
                    )

            frame_data = go.Frame(
                data=[
                    go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=positions[:, 2],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=color,
                            symbol=symbol,
                            opacity=0.8,
                            line=dict(width=1, color="black"),
                        ),
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        name=f"Frame {frame_indices[i]}",
                    )
                ],
                name=str(i),
                layout=go.Layout(title=f"帧 {frame_indices[i]}: {description}"),
            )
            animation_frames.append(frame_data)

        # 创建初始图 - 使用第一帧的信息
        first_frame = frames_data[0]
        first_positions = first_frame["positions"]
        first_step_type = first_frame.get("step_type", "base_state")

        # 初始颜色
        if first_step_type == "base_state":
            initial_color = "green"
            initial_symbol = "circle"
        else:
            initial_color = "blue"
            initial_symbol = "diamond"

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=first_positions[:, 0],
                    y=first_positions[:, 1],
                    z=first_positions[:, 2],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=initial_color,
                        symbol=initial_symbol,
                        opacity=0.8,
                        line=dict(width=1, color="black"),
                    ),
                    name="原子",
                )
            ],
            frames=animation_frames,
        )

        # 添加播放按钮和增强的滑块
        fig.update_layout(
            title="弹性形变轨迹动画 - 物理过程可视化",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                aspectmode="cube",
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.1,
                    y=1.02,
                    buttons=[
                        dict(
                            label="▶ 播放",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=duration, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ 暂停",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(prefix="当前帧: "),
                    len=0.9,
                    x=0.1,
                    y=0.02,
                    steps=[
                        dict(
                            args=[
                                [str(i)],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                ),
                            ],
                            label=self._create_slider_label(frames_data[i]),
                            method="animate",
                        )
                        for i in range(len(animation_frames))
                    ],
                )
            ],
        )

        # 保存文件
        fig.write_html(output_html)
        logger.info(f"保存交互式动画到: {output_html}")

        return output_html

    def _create_slider_label(self, frame_data):
        """
        为滑块创建有意义的标签 - 修复索引对应问题

        Parameters
        ----------
        frame_data : dict
            帧数据

        Returns
        -------
        str
            滑块标签
        """
        step_type = frame_data.get("step_type", "unknown")
        strain_value = frame_data.get("strain_value", 0.0)

        # 确保step_type是字符串而不是字节对象
        if isinstance(step_type, bytes):
            step_type = step_type.decode("utf-8")
        step_type = str(step_type)

        step_mapping = {
            "base_state": "Base",
            "before_internal_relax": f"Deform{strain_value:.3f}",
            "after_internal_relax": f"Relax{strain_value:.3f}",
        }
        return step_mapping.get(step_type, f"{step_type}")

    def plot_energy_evolution(
        self, trajectory_file: str, save_file: str | None = None
    ) -> go.Figure:
        """
        绘制能量演化图

        Parameters
        ----------
        trajectory_file : str
            轨迹文件
        save_file : str, optional
            保存文件名

        Returns
        -------
        fig : plotly.graph_objects.Figure
            能量演化图
        """
        reader = TrajectoryReader(trajectory_file)
        reader.open()

        # 读取所有能量数据
        energies = []
        times = []

        for i in range(reader.n_frames):
            frame = reader.read_frame(i)
            if "energy" in frame:
                energies.append(frame["energy"])
                times.append(frame.get("time", i))

        reader.close()

        # 创建图
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=times,
                y=energies,
                mode="lines+markers",
                name="Total Energy",
                line=dict(color="blue", width=2),
                marker=dict(size=4),
            )
        )

        fig.update_layout(
            title="Energy Evolution",
            xaxis_title="Time (ps)" if times[0] != 0 else "Frame",
            yaxis_title="Energy (eV)",
            template=self.theme,
            hovermode="x unified",
        )

        if save_file:
            if save_file.endswith(".html"):
                fig.write_html(save_file)
            else:
                fig.write_image(save_file)
            logger.info(f"保存能量演化图到: {save_file}")

        return fig

    def plot_stress_strain_interactive(
        self,
        strains: np.ndarray,
        stresses: np.ndarray,
        components: list[str] | None = None,
        title: str = "Stress-Strain Relationship",
        save_html: str | None = None,
    ) -> go.Figure:
        """
        创建交互式应力-应变关系图

        Parameters
        ----------
        strains : np.ndarray
            应变数据 (N, 6)
        stresses : np.ndarray
            应力数据 (N, 6)
        components : list, optional
            分量名称
        title : str
            图标题
        save_html : str, optional
            保存HTML文件

        Returns
        -------
        fig : plotly.graph_objects.Figure
            交互式图
        """
        if components is None:
            components = ["11", "22", "33", "23", "13", "12"]

        fig = make_subplots(
            rows=2, cols=3, subplot_titles=[f"Component {c}" for c in components]
        )

        for i, comp in enumerate(components):
            row = i // 3 + 1
            col = i % 3 + 1

            # 散点
            fig.add_trace(
                go.Scatter(
                    x=strains[:, i],
                    y=stresses[:, i],
                    mode="markers",
                    marker=dict(size=8, color=self.default_colors[i], opacity=0.7),
                    name=f"σ{comp}",
                    legendgroup=comp,
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

            # 拟合线
            if len(strains[:, i]) > 1:
                coeffs = np.polyfit(strains[:, i], stresses[:, i], 1)
                fit_line = np.poly1d(coeffs)
                x_fit = np.linspace(strains[:, i].min(), strains[:, i].max(), 100)
                y_fit = fit_line(x_fit)

                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        line=dict(color=self.default_colors[i], width=2, dash="dash"),
                        name=f"Fit (slope={coeffs[0]:.1f})",
                        legendgroup=comp,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        fig.update_xaxes(title_text="Strain")
        fig.update_yaxes(title_text="Stress (GPa)")

        fig.update_layout(title=title, template=self.theme, height=600, showlegend=True)

        if save_html:
            fig.write_html(save_html)
            logger.info(f"保存应力-应变图到: {save_html}")

        return fig

    def create_trajectory_video(
        self,
        trajectory_file: str,
        output_video: str,
        fps: int = 30,
        skip: int = 1,
        dpi: int = 100,
        figsize: tuple[int, int] = (10, 8),
    ):
        """
        创建轨迹视频文件（MP4/GIF）

        Parameters
        ----------
        trajectory_file : str
            轨迹文件
        output_video : str
            输出视频文件（.mp4或.gif）
        fps : int
            帧率
        skip : int
            跳帧数
        dpi : int
            分辨率
        figsize : tuple
            图像大小
        """
        reader = TrajectoryReader(trajectory_file)
        reader.open()

        info = reader.get_trajectory_info()
        n_frames = info["n_frames"]

        # 创建图形
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # 读取第一帧确定范围
        first_frame = reader.read_frame(0)
        positions = first_frame["positions"]

        # 设置坐标范围
        max_range = np.ptp(positions, axis=0).max()
        mid_points = np.mean(positions, axis=0)

        ax.set_xlim(mid_points[0] - max_range / 2, mid_points[0] + max_range / 2)
        ax.set_ylim(mid_points[1] - max_range / 2, mid_points[1] + max_range / 2)
        ax.set_zlim(mid_points[2] - max_range / 2, mid_points[2] + max_range / 2)

        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")

        # 初始散点
        scatter = ax.scatter([], [], [], c="b", s=50)

        def init():
            scatter._offsets3d = ([], [], [])
            return (scatter,)

        def update(frame_idx):
            actual_idx = frame_idx * skip
            if actual_idx >= n_frames:
                actual_idx = n_frames - 1

            frame = reader.read_frame(actual_idx)
            positions = frame["positions"]

            scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

            ax.set_title(f"Frame {actual_idx}")

            return (scatter,)

        # 创建动画
        n_animation_frames = n_frames // skip
        anim = FuncAnimation(
            fig,
            update,
            frames=n_animation_frames,
            init_func=init,
            blit=False,
            interval=1000 / fps,
        )

        # 保存视频
        if output_video.endswith(".mp4"):
            writer = FFMpegWriter(fps=fps, metadata=dict(artist="ThermoElasticSim"))
            anim.save(output_video, writer=writer, dpi=dpi)
        elif output_video.endswith(".gif"):
            writer = PillowWriter(fps=fps)
            anim.save(output_video, writer=writer, dpi=dpi)
        else:
            raise ValueError(f"不支持的视频格式: {output_video}")

        plt.close(fig)
        reader.close()

        logger.info(f"保存轨迹视频到: {output_video}")

    def plot_volume_evolution(
        self, trajectory_file: str, save_file: str | None = None
    ) -> go.Figure:
        """
        绘制体积演化图

        Parameters
        ----------
        trajectory_file : str
            轨迹文件
        save_file : str, optional
            保存文件名

        Returns
        -------
        fig : plotly.graph_objects.Figure
            体积演化图
        """
        reader = TrajectoryReader(trajectory_file)
        reader.open()

        volumes = []
        times = []

        for i in range(reader.n_frames):
            frame = reader.read_frame(i)
            if "volume" in frame:
                volumes.append(frame["volume"])
            elif "box" in frame:
                box = frame["box"]
                volume = np.linalg.det(box)
                volumes.append(volume)
            else:
                continue

            times.append(frame.get("time", i))

        reader.close()

        # 创建图
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=times,
                y=volumes,
                mode="lines+markers",
                name="Volume",
                line=dict(color="green", width=2),
                marker=dict(size=4),
            )
        )

        fig.update_layout(
            title="Volume Evolution",
            xaxis_title="Time (ps)" if times[0] != 0 else "Frame",
            yaxis_title="Volume (Å³)",
            template=self.theme,
            hovermode="x unified",
        )

        if save_file:
            if save_file.endswith(".html"):
                fig.write_html(save_file)
            else:
                fig.write_image(save_file)
            logger.info(f"保存体积演化图到: {save_file}")

        return fig


# 便捷函数
def quick_visualize_trajectory(
    trajectory_file: str, output_dir: str = "./visualization"
):
    """
    快速可视化轨迹文件

    生成一系列标准可视化图表。

    Parameters
    ----------
    trajectory_file : str
        轨迹文件
    output_dir : str
        输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vis = ModernVisualizer()

    # 生成能量演化图
    vis.plot_energy_evolution(
        trajectory_file, save_file=str(output_path / "energy_evolution.html")
    )

    # 生成体积演化图
    vis.plot_volume_evolution(
        trajectory_file, save_file=str(output_path / "volume_evolution.html")
    )

    # 生成交互式动画
    vis.create_trajectory_animation_plotly(
        trajectory_file, str(output_path / "trajectory_animation.html")
    )

    # 生成视频
    vis.create_trajectory_video(
        trajectory_file, str(output_path / "trajectory.gif"), fps=10
    )

    logger.info(f"可视化文件已保存到: {output_dir}")
