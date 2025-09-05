"""Elastic Wave 场景流水线（阶段A：解析计算）

从YAML配置读取材料与密度，调用
:class:`thermoelasticsim.elastic.wave.ElasticWaveAnalyzer` 计算标准方向
的纵横波速，并将结果保存为JSON与CSV，便于教学展示与后续对比。

配置示例（examples/modern_yaml/elastic_wave.yaml）::

    scenario: elastic_wave
    material: { symbol: Al, structure: fcc }
    wave: { density: 2.70 }
    run: { name: elastic_wave_demo }
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
from typing import Any

from ...elastic import get_material_by_symbol
from ...elastic.wave import ElasticWaveAnalyzer
from ...elastic.wave.visualization import plot_polar_plane, plot_velocity_surface_3d


def _infer_density(symbol: str) -> float | None:
    """
    为常见材料提供默认密度值。

    Parameters
    ----------
    symbol : str
        材料元素符号（如'Al', 'Cu', 'C', 'Au'）。

    Returns
    -------
    float or None
        材料密度（单位：g/cm³）。未知材料返回None，
        此时需要用户在YAML中显式提供wave.density。

    Notes
    -----
    仅为教学演示提供常识性密度值。生产环境应使用
    实验测量值或从材料数据库获取。

    支持的材料：
    - Al: 2.70 g/cm³（铝）
    - Cu: 8.96 g/cm³（铜）
    - C: 3.51 g/cm³（金刚石）
    - Au: 19.32 g/cm³（金）
    """
    table = {
        "Al": 2.70,
        "Cu": 8.96,
        "C": 3.51,  # diamond
        "Au": 19.32,
    }
    return table.get(symbol)


def run_elastic_wave_pipeline(
    cfg, outdir: str, material_symbol: str, potential_kind: str | None = None
) -> dict[str, Any]:
    """
    运行弹性波解析计算流水线（阶段A）。

    Parameters
    ----------
    cfg : ConfigManager
        CLI侧配置管理器。
    outdir : str
        输出目录。
    material_symbol : str
        材料元素符号，如 "Al"、"Cu"、"C"。
    potential_kind : str, optional
        势函数类型（此场景未使用，仅占位保持签名一致）。

    Returns
    -------
    dict
        包含计算结果与输出路径的字典。
    """
    mat = get_material_by_symbol(material_symbol)
    if mat is None:
        raise ValueError(f"未知材料: {material_symbol}")

    # 读取密度，优先取YAML，缺省时按常识表推断
    # 优先从YAML读取；无则从材料参数推导；再退化到常识表
    rho = cfg.get("wave.density", None)
    if rho is None:
        try:
            rho = mat.theoretical_density
        except Exception:
            rho = _infer_density(material_symbol)
    if rho is None:
        raise ValueError("未提供密度，且无法从材料推断。请设置 wave.density")

    C11 = float(mat.literature_elastic_constants["C11"])  # GPa
    C12 = float(mat.literature_elastic_constants["C12"])  # GPa
    C44 = float(mat.literature_elastic_constants["C44"])  # GPa

    analyzer = ElasticWaveAnalyzer(C11=C11, C12=C12, C44=C44, density=float(rho))
    report = analyzer.generate_report()

    # 保存JSON
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, "wave_velocities.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 保存CSV（扁平化）
    csv_path = os.path.join(outdir, "wave_velocities.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["direction", "v_longitudinal", "v_transverse1", "v_transverse2"]
        )  # noqa: E501
        for d, r in report.items():
            writer.writerow([d, r["longitudinal"], r["transverse1"], r["transverse2"]])

    # 友好打印结果到日志/屏幕（中文提示，速度单位km/s）
    log = logging.getLogger(__name__)
    log.info("弹性波速结果 (单位: km/s)：")
    for d, r in report.items():
        log.info(
            f"方向 {d}: v_L={r['longitudinal']:.3f}, v_T1={r['transverse1']:.3f}, v_T2={r['transverse2']:.3f}"
        )

    result = {
        "material": material_symbol,
        "density": float(rho),
        "elastic_constants": {"C11": C11, "C12": C12, "C44": C44},
        "report": report,
        "artifacts": {"json": json_path, "csv": csv_path},
    }

    # 可视化（极图）
    vis_cfg = cfg.get("wave.visualization", {}) or {}
    if bool(vis_cfg.get("enabled", True)):
        planes = vis_cfg.get("planes", None)
        if isinstance(planes, list | tuple) and len(planes) > 0:
            polar_paths: list[str] = []
            for pl in planes:
                plane = str(pl)
                n_angles = int(vis_cfg.get("n_angles", 360))
                dpi = int(vis_cfg.get("dpi", 300))
                # 多平面时忽略用户自定义 output，使用按平面命名避免覆盖
                outname = f"analytic_anisotropy_{plane}.png"
                outpath = os.path.join(outdir, outname)
                polar_paths.append(
                    plot_polar_plane(
                        analyzer,
                        plane=plane,
                        n_angles=n_angles,
                        outpath=outpath,
                        dpi=dpi,
                    )
                )
            log.info(f"已生成各向异性极图: {polar_paths}")
            result["artifacts"]["polar"] = polar_paths
        else:
            plane = str(vis_cfg.get("plane", "001"))
            n_angles = int(vis_cfg.get("n_angles", 360))
            dpi = int(vis_cfg.get("dpi", 300))
            outname = str(vis_cfg.get("output", f"analytic_anisotropy_{plane}.png"))
            if not outname.startswith("analytic_"):
                outname = "analytic_" + outname
            outpath = os.path.join(outdir, outname)
            polar_path = plot_polar_plane(
                analyzer, plane=plane, n_angles=n_angles, outpath=outpath, dpi=dpi
            )
            log.info(f"已生成各向异性极图: {polar_path}")
            result["artifacts"]["polar"] = polar_path

    # 可选3D速度面
    surf_cfg = vis_cfg.get("surface3d", {}) or {}
    if bool(surf_cfg.get("enabled", False)):
        # 压低第三方库在控制台的INFO输出
        for logger_name in (
            "kaleido",
            "kaleido.kaleido",
            "kaleido._kaleido_tab",
            "choreographer",
            "choreographer.browser_async",
            "choreographer.browsers.chromium",
            "choreographer.utils._tmpfile",
            "asyncio",
        ):
            with contextlib.suppress(Exception):
                logging.getLogger(logger_name).setLevel(
                    logging.CRITICAL if logger_name == "asyncio" else logging.WARNING
                )
        n_theta = int(surf_cfg.get("n_theta", 60))
        n_phi = int(surf_cfg.get("n_phi", 120))
        export_png = bool(surf_cfg.get("export_png", False))
        # 一次性输出 L/Tmin/Tmax 三个HTML（默认文件名）
        modes = ("L", "Tmin", "Tmax")
        html_paths: dict[str, str | None] = {}
        png_paths: dict[str, str | None] = {}
        for m in modes:
            html_path, png_path = plot_velocity_surface_3d(
                analyzer,
                mode=m,
                n_theta=n_theta,
                n_phi=n_phi,
                output_html=os.path.join(outdir, f"analytic_velocity_surface_{m}.html"),
                output_png=(
                    os.path.join(outdir, f"analytic_velocity_surface_{m}.png")
                    if export_png
                    else None
                ),
            )
            html_paths[m] = html_path
            png_paths[m] = png_path
        result.setdefault("artifacts", {})["surface3d"] = {
            "html": html_paths,
            "png": png_paths,
        }
        if export_png:
            log.info(
                f"已生成3D速度面: HTML={list(html_paths.values())}, PNG={list(png_paths.values())}"
            )
        else:
            log.info(f"已生成3D速度面(HTML): {list(html_paths.values())}")

    return result
