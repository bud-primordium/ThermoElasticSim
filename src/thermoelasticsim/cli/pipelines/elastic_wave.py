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
from ...elastic.wave.dynamics import (
    DynamicsConfig,
    WaveExcitation,
    simulate_plane_wave_mvp,
)
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

    # =============== Phase B（MVP）：平面波MD传播（可选） ===============
    dyn_cfg = cfg.get("wave.dynamics", {}) or {}
    if bool(dyn_cfg.get("enabled", False)):
        log.info("开始运行弹性波MD传播（MVP）…（仅[100]方向）")
        # 解析配置
        sc = tuple(dyn_cfg.get("supercell", [64, 12, 12]))
        dt = float(dyn_cfg.get("dt", dyn_cfg.get("timestep", 0.5)))
        steps = int(dyn_cfg.get("steps", 40000))
        sample_every = int(dyn_cfg.get("sample_every", 50))
        direction = str(dyn_cfg.get("direction", "x"))
        polarization = str(dyn_cfg.get("polarization", "L"))
        n_waves = int(dyn_cfg.get("n_waves", 2))
        amp_v = float(dyn_cfg.get("amplitude_velocity", 1e-4))
        amp_u = dyn_cfg.get("amplitude_displacement", None)
        amp_u = None if amp_u is None else float(amp_u)

        dyn = DynamicsConfig(
            supercell=sc,
            dt_fs=dt,
            steps=steps,
            sample_every=sample_every,
            direction=direction,  # MVP: 仅支持 'x'
            polarization=polarization,  # 'L'/'Ty'/'Tz'
            n_waves=n_waves,
            amplitude_velocity=amp_v,
            amplitude_displacement=amp_u,
        )
        exc = WaveExcitation(
            direction=direction,
            polarization=polarization,
            n_waves=n_waves,
            amplitude_velocity=amp_v,
            amplitude_displacement=amp_u,
            mode="standing",  # 不依赖解析相速度；默认由源注入驱动
            phase_speed_km_s=None,
        )

        os.makedirs(outdir, exist_ok=True)
        xt_path = os.path.join(outdir, "wave_xt.png")
        # 源注入与记录配置（注意：使用 ConfigManager 的点路径读取嵌套字段）
        use_source = bool(cfg.get("wave.dynamics.source.enabled", True))
        source_slab_fraction = float(
            cfg.get("wave.dynamics.source.slab_fraction", 0.06)
        )
        source_amp_v = float(cfg.get("wave.dynamics.source.amplitude_velocity", 5e-4))
        source_t0 = float(cfg.get("wave.dynamics.source.t0_fs", 200.0))
        source_sigma = float(cfg.get("wave.dynamics.source.sigma_fs", 80.0))
        source_type = str(cfg.get("wave.dynamics.source.type", "gaussian"))
        source_cycles = int(cfg.get("wave.dynamics.source.cycles", 4))
        source_freq_thz = float(cfg.get("wave.dynamics.source.freq_THz", 1.0))

        record_traj = bool(cfg.get("wave.dynamics.record_trajectory.enabled", False))
        traj_file = cfg.get(
            "wave.dynamics.record_trajectory.file",
            os.path.join(outdir, "wave_trajectory.h5"),
        )
        measure_method = str(cfg.get("wave.dynamics.measure.method", "auto"))

        # 将动态配置注入 DynamicsConfig
        dyn.use_source = use_source
        dyn.source_slab_fraction = source_slab_fraction
        dyn.source_amplitude_velocity = source_amp_v
        dyn.source_t0_fs = source_t0
        dyn.source_sigma_fs = source_sigma
        dyn.source_type = source_type
        dyn.source_cycles = source_cycles
        dyn.source_freq_thz = source_freq_thz
        # 探针位置（可调）
        if "detectors" in dyn_cfg:
            dets = list(dyn_cfg.get("detectors", [0.2, 0.7]))
            if isinstance(dets, list | tuple) and len(dets) == 2:
                dyn.detector_frac_a = float(dets[0])
                dyn.detector_frac_b = float(dets[1])
        # 吸收边界（海绵层）
        dyn.absorber_enabled = bool(cfg.get("wave.dynamics.absorber.enabled", False))
        try:
            dyn.absorber_slab_fraction = float(
                cfg.get(
                    "wave.dynamics.absorber.slab_fraction", dyn.absorber_slab_fraction
                )
            )
            dyn.absorber_tau_fs = float(
                cfg.get("wave.dynamics.absorber.tau_fs", dyn.absorber_tau_fs)
            )
            dyn.absorber_profile = str(
                cfg.get("wave.dynamics.absorber.profile", dyn.absorber_profile)
            )
        except Exception:
            pass

        dyn.record_trajectory = record_traj
        dyn.trajectory_file = traj_file
        dyn.measure_method = measure_method
        # 物理最大速度上限（用于互相关约束）
        with contextlib.suppress(Exception):
            dyn.v_max_km_s = float(cfg.get("wave.dynamics.v_max_km_s", dyn.v_max_km_s))

        md_res = simulate_plane_wave_mvp(
            material_symbol=material_symbol,
            dynamics=dyn,
            excitation=exc,
            out_xt_path=xt_path,
        )

        # 与解析（[100]）对比（仅 L 或 T 情况）
        try:
            ana_100 = analyzer.calculate_wave_velocities((1.0, 0.0, 0.0))
            v_ana = None
            pol = polarization.strip()
            if pol == "L":
                v_ana = float(ana_100["longitudinal"])
            elif pol in ("Ty", "Tz"):
                v_ana = float(min(ana_100["transverse1"], ana_100["transverse2"]))
            v_md = md_res.get("velocity_estimate_km_s")
            if v_md is not None and v_ana is not None and v_ana > 0:
                err_pct = abs(v_md - v_ana) / v_ana * 100.0
                md_res["analytic_compare"] = {
                    "v_md_km_s": v_md,
                    "v_ana_km_s": v_ana,
                    "error_percent": err_pct,
                }
        except Exception:
            pass

        dyn_json = os.path.join(outdir, "wave_dynamics.json")
        try:
            with open(dyn_json, "w", encoding="utf-8") as f:
                json.dump(md_res, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        result.setdefault("artifacts", {})["dynamics"] = {
            "xt": xt_path,
            "json": dyn_json,
            "velocity_km_s": md_res.get("velocity_estimate_km_s"),
        }
        v_est = md_res.get("velocity_estimate_km_s")
        if v_est is not None:
            if "analytic_compare" in md_res:
                ap = md_res["analytic_compare"]["error_percent"]
                va = md_res["analytic_compare"]["v_ana_km_s"]
                log.info(
                    f"MD 波速估计 ~ {v_est:.2f} km/s；解析[{polarization or 'L'}] ~ {va:.2f} km/s；误差 ~ {ap:.1f}%"
                )
            else:
                log.info(f"MD 波速估计 ~ {v_est:.2f} km/s（互相关，x=1/4L→3/4L）")
        else:
            reason = None
            with contextlib.suppress(Exception):
                reason = md_res.get("xcorr_info", {}).get("reason", None)
            if reason:
                log.warning(f"MD 波速估计失败（互相关失败原因: {reason}）")
            else:
                log.warning("MD 波速估计失败（互相关未找到有效峰值）")

        # 生成GIF（若记录了轨迹）
        # 优先使用模拟返回的轨迹文件路径（更可靠，避免与配置不一致）
        traj_path_effective = md_res.get("trajectory_file", traj_file)
        if record_traj and traj_path_effective and os.path.exists(traj_path_effective):
            try:
                from ...utils.modern_visualization import ModernVisualizer

                vis = ModernVisualizer()
                gif_path = os.path.join(outdir, "wave_trajectory.gif")
                vis.create_trajectory_video(
                    traj_path_effective, gif_path, fps=12, dpi=120
                )
                result.setdefault("artifacts", {})["trajectory"] = {
                    "h5": traj_path_effective,
                    "gif": gif_path,
                }
                log.info(f"已生成轨迹GIF: {gif_path}")
            except Exception as e:
                log.warning(f"轨迹GIF生成失败: {e}")
        elif record_traj:
            log.warning(f"未找到轨迹文件，跳过GIF生成。期望: {traj_path_effective}")

    return result
