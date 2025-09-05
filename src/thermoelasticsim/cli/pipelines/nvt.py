"""NVT 教学场景流水线

支持 Langevin 与 Nose–Hoover 链（NHC）、Andersen、Berendsen 四种恒温器。
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ...md.schemes import (
    AndersenNVTScheme,
    BerendsenNVTScheme,
    LangevinNVTScheme,
    NoseHooverNVTScheme,
)
from .common import assign_maxwell, build_cell, get_material_by_spec, make_potential


def run_nvt_pipeline(
    cfg, outdir: str, material_symbol: str, potential_kind: str
) -> None:
    """运行 NVT 场景。

    Parameters
    ----------
    cfg : ConfigManager
        配置对象。
    outdir : str
        输出目录。
    material_symbol : str
        材料符号（如 'Al'、'Cu'、'C'）。
    potential_kind : str
        势函数标识（如 'EAM_Al1'）。
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

    nvt_cfg = cfg.get("nvt", {})
    nvt_type = str(nvt_cfg.get("type", "langevin")).lower()
    dt = float(nvt_cfg.get("dt", cfg.get("md.timestep", 0.5)))
    steps = int(nvt_cfg.get("steps", 10000))
    sample_every = int(nvt_cfg.get("sample_every", 50))

    if nvt_type == "langevin":
        friction = float(nvt_cfg.get("friction", 3.0))
        scheme = LangevinNVTScheme(target_temperature=T0, friction=friction)
    elif nvt_type == "nhc":
        tdamp = float(nvt_cfg.get("tdamp", 50.0))
        tchain = int(nvt_cfg.get("tchain", 3))
        tloop = int(nvt_cfg.get("tloop", 1))
        scheme = NoseHooverNVTScheme(
            target_temperature=T0, tdamp=tdamp, tchain=tchain, tloop=tloop
        )
    elif nvt_type == "andersen":
        collision_frequency = float(nvt_cfg.get("collision_frequency", 0.01))
        scheme = AndersenNVTScheme(
            target_temperature=T0, collision_frequency=collision_frequency
        )
    elif nvt_type == "berendsen":
        tau = float(nvt_cfg.get("tau", 100.0))
        scheme = BerendsenNVTScheme(target_temperature=T0, tau=tau)
    else:
        raise ValueError(f"未知NVT类型: {nvt_type}")

    rows = []
    logger = logging.getLogger(__name__)
    for s in range(steps):
        scheme.step(cell, pot, dt)
        if s % sample_every == 0:
            T = cell.calculate_temperature()
            KE = sum(
                0.5 * a.mass * float(np.dot(a.velocity, a.velocity)) for a in cell.atoms
            )
            PE = float(pot.calculate_energy(cell))
            rows.append(
                {
                    "step": s,
                    "time_ps": s * dt / 1000.0,
                    "T_K": T,
                    "KE_eV": KE,
                    "PE_eV": PE,
                    "E_eV": KE + PE,
                }
            )
        if s and (s % max(1, steps // 5) == 0):
            logger.info(f"NVT({nvt_type}) {s:6d}/{steps}: T={rows[-1]['T_K']:.1f} K")

    # 写CSV
    csv_path = os.path.join(outdir, "thermo.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            import csv as _csv

            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # 温度曲线
    try:
        import matplotlib.pyplot as plt

        t = [r["time_ps"] for r in rows]
        T = [r["T_K"] for r in rows]
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
        ax.plot(t, T, label="Temperature (K)")
        ax.set_xlabel("time (ps)")
        ax.set_ylabel("T (K)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(
            os.path.join(outdir, "temperature_evolution.png"),
            dpi=300,
            bbox_inches="tight",
        )
    except Exception:
        pass
    # 统计与RMSE
    try:
        import numpy as _np

        T_arr = _np.array([r["T_K"] for r in rows], dtype=float)
        T_mean = float(_np.mean(T_arr)) if T_arr.size else float("nan")
        T_std = float(_np.std(T_arr)) if T_arr.size else float("nan")
        T_rmse = (
            float(_np.sqrt(_np.mean((T_arr - T0) ** 2))) if T_arr.size else float("nan")
        )
        logging.getLogger(__name__).info(
            f"NVT({nvt_type}) 完成: 步数={steps}, 采样间隔={sample_every}, 输出: {outdir}"
        )
        logging.getLogger(__name__).info(
            f"统计: <T> = {T_mean:.1f} ± {T_std:.1f} K; RMSE={T_rmse:.1f} K"
        )
    except Exception:
        logging.getLogger(__name__).info(
            f"NVT({nvt_type}) 完成: 步数={steps}, 采样间隔={sample_every}, 输出: {outdir}"
        )
