#!/usr/bin/env python3
"""
弹性波各向异性可视化

提供在指定晶面内采样传播方向，计算纵/横波速并绘制极坐标图的工具。

功能
----
- `sample_plane_directions`：在给定面（"001"/"110"/"111"）上等角采样方向
- `compute_velocities_over_directions`：批量计算 v_L, v_T1, v_T2
- `plot_polar_plane`：生成各向异性极图（matplotlib）

说明
----
- 仅依赖 matplotlib，默认使用项目统一的中文字体与Agg后端。
- 角度 θ ∈ [0, 2π)，方向向量 n(θ) = u cosθ + v sinθ，其中 {u, v} 为该平面
  的正交基，法向量为平面的米勒指数方向。
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterable, Sequence
from multiprocessing import Process

import numpy as np

from ...elastic.wave.analytical import ElasticWaveAnalyzer
from ...utils.plot_config import plt


def _canonical_plane_basis(plane: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为指定晶面返回规范化的正交基向量组。

    Parameters
    ----------
    plane : str
        晶面标识符（"001", "110", "111"）。

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        (n, u, v) 三个单位向量：
        - n: 面法向量
        - u, v: 面内正交基向量

    Notes
    -----
    高对称晶面的规范基选择：
    - (001): n=[0,0,1], u=[1,0,0]([100]), v=[0,1,0]([010])
    - (110): n=[1,1,0]/√2, u=[1,-1,0]/√2, v=[0,0,1]([001])
    - (111): n=[1,1,1]/√3, u=[1,-1,0]/√2, v=n×u（归一化）
    """
    plane = plane.strip().strip("[](){}")
    if plane == "001":
        n = np.array([0.0, 0.0, 1.0])
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        return n, u, v
    if plane == "110":
        n = np.array([1.0, 1.0, 0.0])
        n = n / np.linalg.norm(n)
        u = np.array([1.0, -1.0, 0.0])
        u = u / np.linalg.norm(u)
        v = np.array([0.0, 0.0, 1.0])
        return n, u, v
    if plane == "111":
        n = np.array([1.0, 1.0, 1.0])
        n = n / np.linalg.norm(n)
        u = np.array([1.0, -1.0, 0.0])
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        return n, u, v
    # 回退到通用算法
    n = _plane_normal(plane)
    u, v = _orthonormal_basis_for_plane(n)
    return n, u, v


def _plane_normal(plane: str) -> np.ndarray:
    """将平面标识（如"001"）解析为单位法向量。

    Parameters
    ----------
    plane : str
        平面标识，仅支持 "001"、"110"、"111"。

    Returns
    -------
    ndarray, shape (3,)
        单位法向量。
    """
    plane = plane.strip().strip("[](){}")
    if plane not in {"001", "110", "111"}:
        raise ValueError(f"不支持的平面: {plane}")
    h, k, l_idx = (int(ch) for ch in plane)
    v = np.array([h, k, l_idx], dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("非法平面索引")
    return v / n


def _orthonormal_basis_for_plane(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    为给定法向量的平面构造一组正交基向量。

    Parameters
    ----------
    normal : ndarray
        平面法向量（非零）。

    Returns
    -------
    tuple[ndarray, ndarray]
        (u, v) 两个单位正交向量，满足 u⊥v⊥normal。

    Notes
    -----
    使用叉积构造正交基，选择合适的参考向量避免平行退化。
    """
    n = normal / np.linalg.norm(normal)
    # 选择与 n 不平行的参考向量
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    un = np.linalg.norm(u)
    if un == 0:
        # 极罕见，换一个参考向量
        ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(n, ref)
        un = np.linalg.norm(u)
    u = u / un
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return u, v


def sample_plane_directions(
    plane: str, n_angles: int = 360
) -> tuple[np.ndarray, np.ndarray]:
    """
    在指定晶面上等角采样传播方向。

    Parameters
    ----------
    plane : str
        平面标识（"001"/"110"/"111"）。
    n_angles : int, optional
        角度采样数（默认360）。

    Returns
    -------
    tuple[ndarray, ndarray]
        (theta, directions)，其中 theta 形状为 (n_angles,)，
        directions 形状为 (n_angles, 3)，均为单位长度方向向量。
    """
    normal, u, v = _canonical_plane_basis(plane)
    thetas = np.linspace(0.0, 2.0 * math.pi, int(n_angles), endpoint=False)
    dirs = np.cos(thetas)[:, None] * u[None, :] + np.sin(thetas)[:, None] * v[None, :]
    # 标准化以去除数值误差
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return thetas, dirs


def compute_velocities_over_directions(
    analyzer: ElasticWaveAnalyzer,
    directions: Iterable[Iterable[float]],
    plane: str | None = None,
) -> dict[str, np.ndarray]:
    """
    批量计算多个方向上的纵/横波速。

    Parameters
    ----------
    analyzer : ElasticWaveAnalyzer
        解析计算器实例。
    directions : Iterable[Iterable[float]]
        多个单位方向向量。

    Returns
    -------
    dict
        包含 'vL', 'vT1', 'vT2' 三个键，对应等长的速度数组（km/s）。
    """
    vL, vT1, vT2 = [], [], []
    vT_para, vT_perp = [], []
    use_classify = plane is not None
    n_plane = None
    if use_classify:
        n_plane = _canonical_plane_basis(plane)[0]

    for n in directions:
        r = analyzer.calculate_wave_velocities(n)
        vL.append(r["longitudinal"])
        # 横波按升序返回，保持一致
        v1 = r["transverse1"]
        v2 = r["transverse2"]
        vT1.append(v1)
        vT2.append(v2)
        if use_classify:
            pol = r["polarizations"]
            e1 = np.asarray(pol["transverse1"], dtype=float)
            e2 = np.asarray(pol["transverse2"], dtype=float)
            # 计算与平面法向的夹角，法向分量更大者为 T_perp
            d1 = abs(float(np.dot(e1, n_plane)))
            d2 = abs(float(np.dot(e2, n_plane)))
            if d1 >= d2:
                vT_perp.append(v1)
                vT_para.append(v2)
            else:
                vT_perp.append(v2)
                vT_para.append(v1)

    result = {
        "vL": np.asarray(vL, dtype=float),
        "vT1": np.asarray(vT1, dtype=float),
        "vT2": np.asarray(vT2, dtype=float),
    }
    if use_classify:
        result["vT_para"] = np.asarray(vT_para, dtype=float)
        result["vT_perp"] = np.asarray(vT_perp, dtype=float)
    return result


def plot_polar_plane(
    analyzer: ElasticWaveAnalyzer,
    plane: str = "001",
    n_angles: int = 360,
    outpath: str | None = None,
    dpi: int = 300,
    annotate_hkls: Sequence[Sequence[int]] | None = None,
) -> str:
    """
    绘制给定晶面内的声速各向异性极图。

    Parameters
    ----------
    analyzer : ElasticWaveAnalyzer
        解析计算器实例。
    plane : str, optional
        平面标识（"001"/"110"/"111"），默认"001"。
    n_angles : int, optional
        角度采样点数，默认360。
    outpath : str, optional
        输出文件路径；若为None则使用当前目录下的 'anisotropy_polar.png'。
    dpi : int, optional
        保存DPI，默认300。

    Returns
    -------
    str
        生成的图像路径。
    """
    theta, dirs = sample_plane_directions(plane, n_angles)
    v = compute_velocities_over_directions(analyzer, dirs, plane=plane)

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta, v["vL"], label="纵波 v_L", color="#1f77b4", lw=1.8)
    if "vT_para" in v and "vT_perp" in v:
        ax.plot(
            theta, v["vT_perp"], label="横波 v_T⊥ (法向偏振)", color="#2ca02c", lw=1.4
        )
        ax.plot(
            theta, v["vT_para"], label="横波 v_T∥ (面内偏振)", color="#ff7f0e", lw=1.4
        )
    else:
        ax.plot(theta, v["vT1"], label="横波 v_T1", color="#ff7f0e", lw=1.4)
        ax.plot(theta, v["vT2"], label="横波 v_T2", color="#2ca02c", lw=1.4)
    ax.set_title(f"{plane} 平面声速各向异性 (km/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.15))

    # 注记指定方向（仅对位于该平面的方向有效）
    if annotate_hkls is None:
        if plane == "001":
            annotate_hkls = ([1, 0, 0], [1, 1, 0])  # [100], [110]
        elif plane == "110":
            annotate_hkls = ([0, 0, 1], [1, -1, 0])  # [001], [1-10]
        elif plane == "111":
            annotate_hkls = ([1, -1, 0], [1, 1, -2])  # [1-10], [11-2]
    if annotate_hkls:
        n, u, v_basis = _canonical_plane_basis(plane)
        for hkl in annotate_hkls:
            d = np.array(hkl, dtype=float)
            d = d / (np.linalg.norm(d) or 1.0)
            d_proj = d - np.dot(d, n) * n
            norm = np.linalg.norm(d_proj)
            if norm < 1e-8:
                continue  # 方向不在平面内，跳过
            d_proj /= norm
            ang = math.atan2(np.dot(d_proj, v_basis), np.dot(d_proj, u))
            if ang < 0:
                ang += 2 * math.pi
            # 画径向线与文本（靠外半径）
            rmax = float(np.nanmax([v["vL"].max(), v["vT1"].max(), v["vT2"].max()]))
            ax.plot([ang, ang], [0, rmax * 1.02], color="#444444", lw=0.8, alpha=0.5)
            ax.text(
                ang,
                rmax * 1.08,
                f"[{hkl[0]}{hkl[1]}{hkl[2]}]",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#2c2c2c",
            )

    # 信息框：材料参数与物理说明
    C11, C12, C44 = analyzer.C11, analyzer.C12, analyzer.C44
    rho = analyzer.density
    A = 2.0 * C44 / (C11 - C12) if (C11 - C12) != 0 else np.inf

    # 计算各向异性程度说明
    aniso_desc = (
        "各向同性"
        if abs(A - 1.0) < 0.1
        else ("强各向异性" if abs(A - 1.0) > 0.5 else "弱各向异性")
    )

    lines = [
        f"C11={C11:.1f} GPa, C12={C12:.1f} GPa, C44={C44:.1f} GPa, ρ={rho:.2f} g/cm³",
        f"各向异性因子 A=2C44/(C11−C12)={A:.2f} ({aniso_desc})",
    ]
    if plane == "001":
        lines.append("(001)平面: 法向偏振横波v_T⊥=√(C44/ρ)恒定，面内横波呈四重对称")
    elif plane == "110":
        lines.append("(110)平面: 纵波沿[110]方向最快，横波呈现明显各向异性")
    elif plane == "111":
        lines.append("(111)平面: 高对称面，横波简并，呈现三重旋转对称")
    # 将说明放置在图下方，避免与雷达图重叠
    fig.text(
        0.02,
        0.02,
        "\n".join(lines),
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa", alpha=0.9),
    )

    out = outpath or "analytic_anisotropy_polar.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


def plot_velocity_surface_3d(
    analyzer: ElasticWaveAnalyzer,
    mode: str = "L",
    n_theta: int = 60,
    n_phi: int = 120,
    output_html: str | None = None,
    output_png: str | None = None,
) -> tuple[str | None, str | None]:
    """
    绘制三维速度面（plotly，可交互）。

    Parameters
    ----------
    analyzer : ElasticWaveAnalyzer
        解析计算器实例。
    mode : {"L", "Tmin", "Tmax"}
        绘制纵波面或两支横波中的较小/较大。
    n_theta, n_phi : int
        角度网格分辨率（θ: [0,π], φ: [0,2π)）。
    output_html : str, optional
        交互版HTML文件路径（推荐）。
    output_png : str, optional
        静态PNG文件路径（需要安装kaleido）。

    Returns
    -------
    (html_path, png_path)
        实际生成的文件路径；若依赖缺失则返回 (None, None)。
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return None, None

    thetas = np.linspace(0.0, math.pi, int(n_theta))
    phis = np.linspace(0.0, 2.0 * math.pi, int(n_phi), endpoint=False)
    T, P = np.meshgrid(thetas, phis, indexing="ij")
    # 方向向量
    nx = np.sin(T) * np.cos(P)
    ny = np.sin(T) * np.sin(P)
    nz = np.cos(T)

    shape = T.shape
    v = np.zeros(shape, dtype=float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            r = analyzer.calculate_wave_velocities((nx[i, j], ny[i, j], nz[i, j]))
            if mode == "L":
                v[i, j] = r["longitudinal"]
            elif mode == "Tmin":
                v[i, j] = min(r["transverse1"], r["transverse2"])
            else:  # Tmax
                v[i, j] = max(r["transverse1"], r["transverse2"])

    # 球半径 = v，转换到直角坐标
    X = v * nx
    Y = v * ny
    Z = v * nz

    # 为避免φ=0与2π处出现开口（seam），在列方向拼接首列副本闭合表面
    X = np.concatenate([X, X[:, :1]], axis=1)
    Y = np.concatenate([Y, Y[:, :1]], axis=1)
    Z = np.concatenate([Z, Z[:, :1]], axis=1)
    v = np.concatenate([v, v[:, :1]], axis=1)

    surface = go.Surface(
        x=X,
        y=Y,
        z=Z,
        surfacecolor=v,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="v (km/s)"),
    )

    # 标注 [100]/[110]/[111] 散点（按当前模式的半径）
    def _v_mode_at(dir_vec: np.ndarray) -> float:
        res = analyzer.calculate_wave_velocities(dir_vec)
        if mode == "L":
            return float(res["longitudinal"])
        elif mode == "Tmin":
            return float(min(res["transverse1"], res["transverse2"]))
        else:
            return float(max(res["transverse1"], res["transverse2"]))

    pts = {
        "[100]": np.array([1.0, 0.0, 0.0]),
        "[110]": (np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)),
        "[111]": (np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)),
    }
    xs, ys, zs, texts = [], [], [], []
    for label, nvec in pts.items():
        r_ = _v_mode_at(nvec)
        p = r_ * nvec
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])
        texts.append(label)

    scatter = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text",
        marker=dict(size=5, color="red"),
        text=texts,
        textposition="top center",
        name="高对称点",
    )

    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        title=f"3D 速度面 - {mode}",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    )

    html_path = output_html or f"analytic_velocity_surface_{mode}.html"
    fig.write_html(html_path)

    png_path = None
    if output_png:
        if _write_image_with_timeout(fig, output_png, timeout=30):
            png_path = output_png
        else:
            png_path = None

    return html_path, png_path


def _worker_write_image_figjson(fig_json: str, path: str) -> None:
    try:
        import plotly.io as pio  # type: ignore

        fig = pio.from_json(fig_json)
        fig.write_image(path)
    except Exception:
        pass


def _write_image_with_timeout(fig, output_png: str, timeout: int = 30) -> bool:
    """在子进程中导出PNG，并设置超时，防止阻塞。"""
    try:
        fig_json = fig.to_json()
    except Exception:
        return False

    proc = Process(
        target=_worker_write_image_figjson, args=(fig_json, output_png), daemon=True
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        with contextlib.suppress(Exception):
            proc.terminate()
        return False
    return os.path.exists(output_png) and os.path.getsize(output_png) > 0


__all__ = [
    "sample_plane_directions",
    "compute_velocities_over_directions",
    "plot_polar_plane",
    "plot_velocity_surface_3d",
]
