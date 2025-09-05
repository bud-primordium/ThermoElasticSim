"""NPT（MTK）教学场景流水线

只运行 NPT 预平衡，输出压力/温度时间序列与压力演化图。
"""

from __future__ import annotations

import csv
import logging
import os

import numpy as np

from ...md.schemes import create_mtk_npt_scheme
from .common import assign_maxwell, build_cell, get_material_by_spec, make_potential


def run_npt_pipeline(
    cfg, outdir: str, material_symbol: str, potential_kind: str
) -> None:
    """运行NPT（MTK）教学场景流水线。

    执行NPT预平衡，输出压力/温度时间序列与压力演化图。
    """
    material_cfg = cfg.get("material", None)
    if isinstance(material_cfg, dict):
        mat = get_material_by_spec(
            material_cfg.get("symbol", material_symbol), material_cfg.get("structure")
        )
    else:
        mat = get_material_by_spec(material_symbol, cfg.get("material.structure", None))
    pot = make_potential(potential_kind)

    sc = tuple(cfg.get("supercell", [3, 3, 3]))
    cell = build_cell(mat, sc)
    T0 = float(cfg.get("md.temperature", 300.0))
    assign_maxwell(cell, T0)

    npt = cfg.get("npt", cfg.get("finite_temp.npt", {}))
    P_target = float(cfg.get("md.pressure", 0.0))
    dt = float(npt.get("dt", 0.15))
    tdamp = float(npt.get("tdamp", 50.0))
    pdamp = float(npt.get("pdamp", 150.0))
    steps = int(npt.get("steps", 20000))
    sample_every = int(npt.get("sample_every", 50))
    ma_window_ps = float(npt.get("ma_window_ps", 0.5))
    scheme = create_mtk_npt_scheme(
        T0, P_target, tdamp, pdamp, int(npt.get("tchain", 3)), int(npt.get("pchain", 3))
    )

    t_ps: list[float] = []
    T_hist: list[float] = []
    P_hist: list[float] = []
    logger = logging.getLogger(__name__)
    for s in range(steps):
        scheme.step(cell, pot, dt)
        if s % sample_every == 0:
            t_ps.append(s * dt / 1000.0)
            T_hist.append(cell.calculate_temperature())
            sigma = cell.calculate_stress_tensor(pot)
            P_gpa = (-np.trace(sigma) / 3.0) / 6.2415e-3
            P_hist.append(P_gpa)
        if s and (s % max(1, steps // 5) == 0):
            logger.info(
                f"NPT {s:6d}/{steps}: T={T_hist[-1]:.1f}K, P={P_hist[-1]:+.3f} GPa"
            )

    csv_path = os.path.join(outdir, "pressure.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_ps", "T_K", "P_GPa"])
        for a, b, c in zip(t_ps, T_hist, P_hist, strict=False):
            w.writerow([a, b, c])

    try:
        import matplotlib.pyplot as plt

        t = np.array(t_ps)
        P = np.array(P_hist)
        ma_w = max(1, int(ma_window_ps * 1000.0 / dt / sample_every))
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
        ax.plot(t, P, color="#888", label="Instant P")
        ax.plot(t, P_ma, color="#0072B2", label="MA")
        ax.axhline(
            y=P_target, color="#D55E00", linestyle="--", label=f"Target {P_target} GPa"
        )
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("P (GPa)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(
            os.path.join(outdir, "npt_pressure_evolution.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # 补充温度曲线
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 3.5))
        ax2.plot(t, np.array(T_hist), color="#009E73", label="Temperature (K)")
        ax2.axhline(y=T0, color="#D55E00", linestyle="--", label=f"Target {T0} K")
        ax2.set_xlabel("time (ps)")
        ax2.set_ylabel("T (K)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.savefig(
            os.path.join(outdir, "temperature_evolution.png"),
            dpi=300,
            bbox_inches="tight",
        )
    except Exception:
        pass

    logging.getLogger(__name__).info(
        f"NPT完成: 步数={steps}, 采样间隔={sample_every}, 输出: {outdir}"
    )
