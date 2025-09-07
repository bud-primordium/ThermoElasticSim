#!/usr/bin/env python3
"""
可视化统一接口（Facade）

为教程与CLI提供稳定、简洁的一组顶层可视化函数，
内部委托给现有的 ModernVisualizer / ResponsePlotter 以及
弹性波可视化模块，避免用户理解多个底层模块细节。

设计原则
--------
- 保持现有实现稳定：不改动底层实现，仅做轻量封装；
- 统一命名与参数风格：函数名与参数尽量与教学语境一致；
- 文档友好：提供中文NumPy风格docstring，便于Sphinx渲染；
- 返回值与副作用遵循底层实现，避免语义分歧。
"""

from __future__ import annotations

from collections.abc import Sequence

from thermoelasticsim.elastic.wave import visualization as wave_viz
from thermoelasticsim.elastic.wave.analytical import ElasticWaveAnalyzer
from thermoelasticsim.utils.modern_visualization import ModernVisualizer
from thermoelasticsim.visualization.elastic import ResponsePlotter

# ========================= 结构与轨迹可视化 ========================= #


def plot_structure_3d(cell, **kwargs):
    """
    晶体结构3D可视化（交互式）

    Parameters
    ----------
    cell : Cell
        晶胞对象。
    **kwargs
        透传给 ``ModernVisualizer.plot_structure_3d`` 的参数（如
        ``show_box``, ``show_bonds``, ``bond_cutoff``, ``title``, ``save_html``）。

    Returns
    -------
    plotly.graph_objects.Figure
        交互式3D图对象。
    """
    vis = ModernVisualizer()
    return vis.plot_structure_3d(cell, **kwargs)


def plot_trajectory_animation(trajectory_file: str, output_html: str, **kwargs) -> None:
    """
    生成交互式轨迹动画（HTML）

    Parameters
    ----------
    trajectory_file : str
        轨迹文件路径（HDF5等）。
    output_html : str
        输出HTML路径。
    **kwargs
        透传给 ``ModernVisualizer.create_trajectory_animation_plotly`` 的参数
        （如 ``skip``, ``duration``）。

    Notes
    -----
    本函数无返回值，调用后在 ``output_html`` 写入文件。
    """
    vis = ModernVisualizer()
    vis.create_trajectory_animation_plotly(trajectory_file, output_html, **kwargs)


def plot_energy_evolution(trajectory_file: str, save_file: str | None = None):
    """
    绘制能量随时间的演化曲线

    Parameters
    ----------
    trajectory_file : str
        轨迹文件路径。
    save_file : str, optional
        保存路径（``.html`` 或图片后缀）。为 ``None`` 时不写盘。

    Returns
    -------
    plotly.graph_objects.Figure
        能量演化图对象。
    """
    vis = ModernVisualizer()
    return vis.plot_energy_evolution(trajectory_file, save_file)


def plot_stress_strain_interactive(
    strains,
    stresses,
    components: list[str] | None = None,
    title: str = "Stress–Strain",
    save_html: str | None = None,
):
    """
    交互式应力–应变关系图（多子图）

    Parameters
    ----------
    strains : ndarray, shape (N, 6)
        应变数组（Voigt顺序）。
    stresses : ndarray, shape (N, 6)
        应力数组（Voigt顺序，单位 GPa）。
    components : list of str, optional
        分量标签，默认 ``["11","22","33","23","13","12"]``。
    title : str
        总标题。
    save_html : str, optional
        保存HTML路径；为 ``None`` 时不写盘。

    Returns
    -------
    plotly.graph_objects.Figure
        交互式图对象。
    """
    vis = ModernVisualizer()
    return vis.plot_stress_strain_interactive(
        strains, stresses, components=components, title=title, save_html=save_html
    )


def create_trajectory_video(
    trajectory_file: str,
    output_video: str,
    fps: int = 30,
    skip: int = 1,
    dpi: int = 100,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    创建轨迹视频（MP4/GIF）

    Parameters
    ----------
    trajectory_file : str
        轨迹文件路径。
    output_video : str
        输出视频路径（后缀 ``.mp4`` 或 ``.gif``）。
    fps : int, optional
        帧率，默认 30。
    skip : int, optional
        跳帧采样间隔，默认 1。
    dpi : int, optional
        分辨率，默认 100。
    figsize : tuple[int, int], optional
        图像尺寸，默认 (10, 8)。
    """
    vis = ModernVisualizer()
    vis.create_trajectory_video(
        trajectory_file,
        output_video,
        fps=fps,
        skip=skip,
        dpi=dpi,
        figsize=figsize,
    )


# ========================= 弹性波可视化 ========================= #


def plot_wave_anisotropy(
    analyzer: ElasticWaveAnalyzer,
    plane: str = "001",
    n_angles: int = 360,
    outpath: str | None = None,
    dpi: int = 300,
    annotate_hkls: Sequence[Sequence[int]] | None = None,
) -> str:
    """
    绘制指定晶面内的声速各向异性极图

    Parameters
    ----------
    analyzer : ElasticWaveAnalyzer
        波速解析计算器（Christoffel）。
    plane : {"001","110","111"}
        晶面标识。
    n_angles : int
        等角采样点数。
    outpath : str, optional
        输出图像路径；为 ``None`` 使用默认文件名。
    dpi : int
        保存DPI。
    annotate_hkls : Sequence[Sequence[int]], optional
        在图中标注的高对称方向集合，例如 ``[[1,0,0],[1,1,0]]``。

    Returns
    -------
    str
        生成的图像路径。
    """
    return wave_viz.plot_polar_plane(
        analyzer,
        plane=plane,
        n_angles=n_angles,
        outpath=outpath,
        dpi=dpi,
        annotate_hkls=annotate_hkls,
    )


def plot_wave_anisotropy_from_constants(
    C11: float,
    C12: float,
    C44: float,
    density: float,
    plane: str = "001",
    n_angles: int = 360,
    outpath: str | None = None,
    dpi: int = 300,
    annotate_hkls: Sequence[Sequence[int]] | None = None,
) -> str:
    """
    由弹性常数直接绘制声速各向异性极图

    Parameters
    ----------
    C11, C12, C44 : float
        立方晶系三个独立弹性常数（单位 GPa）。
    density : float
        材料密度（单位 g/cm³）。
    plane, n_angles, outpath, dpi, annotate_hkls
        同 :func:`plot_wave_anisotropy`。

    Returns
    -------
    str
        生成的图像路径。
    """
    ana = ElasticWaveAnalyzer(C11=C11, C12=C12, C44=C44, density=density)
    return plot_wave_anisotropy(
        ana,
        plane=plane,
        n_angles=n_angles,
        outpath=outpath,
        dpi=dpi,
        annotate_hkls=annotate_hkls,
    )


def plot_velocity_surface_3d(
    analyzer: ElasticWaveAnalyzer,
    mode: str = "L",
    n_theta: int = 60,
    n_phi: int = 120,
    output_html: str | None = None,
    output_png: str | None = None,
) -> tuple[str | None, str | None]:
    """
    绘制三维速度面（交互/静态）

    Parameters
    ----------
    analyzer : ElasticWaveAnalyzer
        波速解析计算器。
    mode : {"L","Tmin","Tmax"}
        纵波或两支横波中的较小/较大。
    n_theta, n_phi : int
        角度网格分辨率。
    output_html : str, optional
        交互HTML路径（plotly）。
    output_png : str, optional
        静态PNG路径（需要 kaleido）。

    Returns
    -------
    (str | None, str | None)
        (实际生成的HTML路径, PNG路径)。依赖缺失时可能为 ``None``。
    """
    return wave_viz.plot_velocity_surface_3d(
        analyzer,
        mode=mode,
        n_theta=n_theta,
        n_phi=n_phi,
        output_html=output_html,
        output_png=output_png,
    )


def plot_velocity_surface_3d_from_constants(
    C11: float,
    C12: float,
    C44: float,
    density: float,
    mode: str = "L",
    n_theta: int = 60,
    n_phi: int = 120,
    output_html: str | None = None,
    output_png: str | None = None,
) -> tuple[str | None, str | None]:
    """
    由弹性常数直接绘制三维速度面

    其余参数与 :func:`plot_velocity_surface_3d` 一致。
    """
    ana = ElasticWaveAnalyzer(C11=C11, C12=C12, C44=C44, density=density)
    return plot_velocity_surface_3d(
        ana,
        mode=mode,
        n_theta=n_theta,
        n_phi=n_phi,
        output_html=output_html,
        output_png=output_png,
    )


# ========================= 弹性常数响应图 ========================= #


def plot_c11_c12_combined_response(
    c11_data: list[dict],
    c12_data: list[dict],
    supercell_size: tuple[int, int, int],
    output_path: str,
    slope_override_c11: float | None = None,
    slope_override_c12: float | None = None,
    subtitle_c11: str | None = None,
    subtitle_c12: str | None = None,
) -> str:
    """
    生成 C11/C12 联合应力–应变响应图

    Parameters
    ----------
    c11_data, c12_data : list of dict
        每个元素对应一个采样点，需包含 ``applied_strain``、``measured_stress_GPa``、
        ``optimization_converged`` 等键。
    supercell_size : tuple[int, int, int]
        超胞尺寸，如 ``(3,3,3)``。
    output_path : str
        输出图片路径。
    slope_override_c11, slope_override_c12 : float, optional
        手动覆盖拟合斜率（GPa），用于对比。
    subtitle_c11, subtitle_c12 : str, optional
        子图副标题。

    Returns
    -------
    str
        生成的图像文件名（非完整路径，遵循底层实现）。
    """
    plotter = ResponsePlotter()
    return plotter.plot_c11_c12_combined_response(
        c11_data,
        c12_data,
        supercell_size,
        output_path,
        slope_override_c11=slope_override_c11,
        slope_override_c12=slope_override_c12,
        subtitle_c11=subtitle_c11,
        subtitle_c12=subtitle_c12,
    )


def plot_shear_response(
    detailed_results: list[dict],
    supercell_size: tuple[int, int, int],
    output_path: str,
) -> str:
    """
    生成 C44/C55/C66 剪切应力–应变响应图

    Parameters
    ----------
    detailed_results : list of dict
        每个字典包含一个剪切模式的多点数据（应变/应力/收敛标记等）。
    supercell_size : tuple[int, int, int]
        超胞尺寸，如 ``(3,3,3)``。
    output_path : str
        输出图片路径。

    Returns
    -------
    str
        生成的图像文件名（非完整路径，遵循底层实现）。
    """
    plotter = ResponsePlotter()
    return plotter.plot_shear_response(detailed_results, supercell_size, output_path)


__all__ = [
    # 结构/轨迹
    "plot_structure_3d",
    "plot_trajectory_animation",
    "plot_energy_evolution",
    "plot_stress_strain_interactive",
    "create_trajectory_video",
    # 弹性波
    "plot_wave_anisotropy",
    "plot_wave_anisotropy_from_constants",
    "plot_velocity_surface_3d",
    "plot_velocity_surface_3d_from_constants",
    # 弹性常数响应
    "plot_c11_c12_combined_response",
    "plot_shear_response",
]
