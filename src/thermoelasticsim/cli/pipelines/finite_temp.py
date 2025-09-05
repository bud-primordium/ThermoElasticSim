"""有限温弹性常数场景流水线

预热（Langevin）→ NPT 预平衡（MTK）→ NHC 生产拟合 C11/C12/C44。
产物与 Python 版 benchmark 保持一致：压力演化图、三张拟合图、JSON 汇总。
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable

import numpy as np

from ...elastic import StructureRelaxer
from ...md.propagators import NoseHooverChainPropagator
from ...md.schemes import LangevinNVTScheme, create_mtk_npt_scheme
from ...utils.utils import EV_TO_GPA
from .common import (
    apply_strain,
    assign_maxwell,
    build_cell,
    get_material_by_spec,
    make_potential,
    plot_series,
)


def run_finite_temp_pipeline(
    cfg, outdir: str, material_symbol: str, potential_kind: str
) -> None:
    """运行有限温弹性常数完整流程。"""
    # 材料/势/结构
    material_cfg = cfg.get("material", None)
    if isinstance(material_cfg, dict):
        mat = get_material_by_spec(
            material_cfg.get("symbol", material_symbol), material_cfg.get("structure")
        )
    else:
        mat = get_material_by_spec(material_symbol, cfg.get("material.structure", None))
    pot = make_potential(potential_kind)

    sc = tuple(cfg.get("supercell", [4, 4, 4]))
    cell = build_cell(mat, sc)
    logger = logging.getLogger(__name__)

    # 零温基态（等比例弛豫优先）
    relaxer = StructureRelaxer(
        optimizer_type="L-BFGS",
        optimizer_params={"ftol": 1e-7, "gtol": 1e-6, "maxiter": 5000},
        supercell_dims=sc,
    )
    if not relaxer.uniform_lattice_relax(cell, pot):
        relaxer.full_relax(cell, pot)

    # 1) 预热（Langevin）
    T = float(cfg.get("md.temperature", mat.temperature or 300.0))
    pre = cfg.get("finite_temp.preheat", {})
    pre_dt = float(pre.get("dt", 0.5))
    pre_steps = int(pre.get("steps", 10000))
    friction = float(pre.get("friction", 3.0))
    assign_maxwell(cell, 20.0)
    scheme_pre = LangevinNVTScheme(target_temperature=T, friction=friction)
    for s in range(pre_steps):
        scheme_pre.step(cell, pot, pre_dt)
        if s % max(1, pre_steps // 5) == 0:
            logger.info(
                f"预热 {s:6d}/{pre_steps}: T={cell.calculate_temperature():.1f} K"
            )

    # 2) NPT 预平衡（MTK）
    npt = cfg.get("finite_temp.npt", {})
    P_target = float(cfg.get("md.pressure", 0.0))
    npt_dt = float(npt.get("dt", 0.15))
    tdamp = float(npt.get("tdamp", 50.0))
    pdamp = float(npt.get("pdamp", 150.0))
    npt_steps = int(npt.get("steps", 20000))
    npt_sample = int(npt.get("sample_every", 50))
    ma_window_ps = float(npt.get("ma_window_ps", 0.5))
    scheme_npt = create_mtk_npt_scheme(
        T, P_target, tdamp, pdamp, int(npt.get("tchain", 3)), int(npt.get("pchain", 3))
    )

    t_ps: list[float] = []
    T_hist: list[float] = []
    P_hist: list[float] = []
    V_hist: list[float] = []
    for s in range(npt_steps):
        scheme_npt.step(cell, pot, npt_dt)
        if (s % npt_sample) == 0:
            t_ps.append(s * npt_dt / 1000.0)
            T_hist.append(cell.calculate_temperature())
            sigma = cell.calculate_stress_tensor(pot)
            P_gpa = (-np.trace(sigma) / 3.0) / 6.2415e-3
            P_hist.append(P_gpa)
            V_hist.append(cell.volume)
        if s % max(1, npt_steps // 10) == 0:
            logger.info(
                f"NPT {s:6d}/{npt_steps}: T={T_hist[-1]:.1f}K, P={P_hist[-1]:+.3f} GPa, V={V_hist[-1]:.2f} Å³"
            )

    # 压力演化图
    try:
        import matplotlib.pyplot as plt

        t = np.array(t_ps)
        P = np.array(P_hist)
        ma_w = max(1, int(ma_window_ps * 1000.0 / npt_dt / npt_sample))
        if ma_w > len(P):
            ma_w = 1

        def _moving_avg(x, w):
            if w <= 1:
                return x.copy()
            c = np.convolve(x, np.ones(w), mode="valid") / w
            pad = np.full(w - 1, np.nan)
            return np.concatenate([pad, c])

        P_ma = _moving_avg(P, ma_w)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
        ax.plot(t, P, color="#888", linewidth=1.2, label="瞬时压力")
        ax.plot(
            t,
            P_ma,
            color="#0072B2",
            linewidth=2.0,
            label=f"移动平均 (~{ma_window_ps:.1f} ps)",
        )
        ax.axhline(
            y=P_target,
            color="#D55E00",
            linestyle="--",
            label=f"目标压力 {P_target} GPa",
        )
        ax.set_xlabel("时间 (ps)")
        ax.set_ylabel("压力 (GPa)")
        ax.set_title("MTK-NPT 压力演化")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(
            os.path.join(outdir, "npt_pressure_evolution.png"),
            dpi=300,
            bbox_inches="tight",
        )
    except Exception:
        pass

    # NHC 生产采样并拟合
    sigma_zero = cell.calculate_stress_tensor(pot) * EV_TO_GPA
    nhc_cfg = cfg.get("finite_temp.nhc", {})
    nhc_dt = float(nhc_cfg.get("dt", 0.5))
    nhc_tdamp = float(nhc_cfg.get("tdamp", 50.0))
    pre_steps = int(nhc_cfg.get("pre_steps", 6000))
    prod_steps = int(nhc_cfg.get("prod_steps", 20000))
    sample_every = int(nhc_cfg.get("sample_every", 50))

    def _nhc_series(eq_cell, mode: str, mags: Iterable[float]):
        nhc = NoseHooverChainPropagator(
            target_temperature=T,
            tdamp=nhc_tdamp,
            tchain=int(nhc_cfg.get("tchain", 3)),
            tloop=int(nhc_cfg.get("tloop", 1)),
        )
        for i in range(pre_steps):
            nhc.propagate(eq_cell, nhc_dt)
            if i and (i % max(1, pre_steps // 3) == 0):
                logger.info(
                    f"NHC 预平衡 {i:6d}/{pre_steps}: T={eq_cell.calculate_temperature():.1f} K"
                )
        xs: list[float] = []
        ys: list[float] = []
        es: list[float] = []
        for m in mags:
            dcell = apply_strain(eq_cell, mode, float(m))
            temps: list[float] = []
            sigmas: list[np.ndarray] = []
            for s in range(prod_steps):
                nhc.propagate(dcell, nhc_dt)
                if s % sample_every == 0:
                    temps.append(dcell.calculate_temperature())
                    sigmas.append(dcell.calculate_stress_tensor(pot) * EV_TO_GPA)
                if s and (s % max(1, prod_steps // 4) == 0):
                    logger.info(
                        f"NHC 生产 {mode} {s:6d}/{prod_steps}: T={temps[-1]:.1f} K"
                    )
            sig_avg = np.mean(sigmas, axis=0)
            sig_std = np.std(sigmas, axis=0)
            comp = (0, 0) if mode == "xx" else (1, 2)
            y = float(sig_avg[comp] - sigma_zero[comp])
            e = float(sig_std[comp])
            xs.append(float(m))
            ys.append(y)
            es.append(e)
        if len(xs) >= 2:
            coeffs = np.polyfit(np.array(xs), np.array(ys), 1)
            Cval = float(coeffs[0])
        else:
            coeffs = [0.0, 0.0]
            Cval = float("nan")
        return xs, ys, es, coeffs, Cval

    default_mags = [-0.005, -0.004, -0.003, -0.002, 0.002, 0.003, 0.004, 0.005]
    mags_c11 = cfg.get("finite_temp.strains.C11", default_mags)
    mags_c12 = cfg.get("finite_temp.strains.C12", default_mags)
    mags_c44 = cfg.get("finite_temp.strains.C44", default_mags)

    eq_cell = cell.copy()
    xs11, ys11, es11, coeffs11, C11 = _nhc_series(eq_cell.copy(), "xx", mags_c11)
    xs12, ys12, es12, coeffs12, C12 = _nhc_series(eq_cell.copy(), "xx", mags_c12)
    xs44, ys44, es44, coeffs44, C44 = _nhc_series(eq_cell.copy(), "yz", mags_c44)

    p11 = plot_series(outdir, "C11", xs11, ys11, es11, coeffs11, C11)
    p12 = plot_series(outdir, "C12", xs12, ys12, es12, coeffs12, C12)
    p44 = plot_series(outdir, "C44", xs44, ys44, es44, coeffs44, C44)

    res = {
        "meta": {
            "system": f"{mat.symbol} {mat.structure} {sc[0]}x{sc[1]}x{sc[2]}",
            "T": T,
            "P_target_GPa": P_target,
            "npt_pressure_plot": os.path.join(outdir, "npt_pressure_evolution.png"),
        },
        "C11": {"value": C11, "plot": p11, "coeffs": list(map(float, coeffs11))},
        "C12": {"value": C12, "plot": p12, "coeffs": list(map(float, coeffs12))},
        "C44": {"value": C44, "plot": p44, "coeffs": list(map(float, coeffs44))},
    }
    with open(
        os.path.join(outdir, "finite_temp_elastic_mtk_complete.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    logger.info("-" * 60)
    logger.info(f"C11={C11:.1f} GPa; 图: {p11}")
    logger.info(f"C12={C12:.1f} GPa; 图: {p12}")
    logger.info(f"C44={C44:.1f} GPa; 图: {p44}")
    logger.info("-" * 60)
    logger.info(
        f"NHC 生产采样完成: 预平衡步数={pre_steps}, 生产步数={prod_steps}, 采样间隔={sample_every}"
    )
