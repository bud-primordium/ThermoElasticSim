"""NVE 教学场景流水线（能量守恒演示）"""

from __future__ import annotations

import csv
import logging
import os

import numpy as np

from ...md.schemes import NVEScheme
from .common import assign_maxwell, build_cell, get_material_by_spec, make_potential


def run_nve_pipeline(
    cfg, outdir: str, material_symbol: str, potential_kind: str
) -> None:
    """运行NVE教学场景流水线。

    执行微正则系综模拟，演示能量守恒特性。
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

    dt = float(cfg.get("nve.dt", cfg.get("md.timestep", 0.5)))
    steps = int(cfg.get("nve.steps", 10000))
    sample_every = int(cfg.get("nve.sample_every", 50))

    scheme = NVEScheme()

    rows = []
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
            logging.getLogger(__name__).info(
                f"NVE {s:6d}/{steps}: T={rows[-1]['T_K']:.1f} K"
            )

    if rows:
        with open(
            os.path.join(outdir, "thermo.csv"), "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

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
        # 能量守恒图（分图，避免量级差异影响可读性）
        E = [r["E_eV"] for r in rows]
        KE = [r["KE_eV"] for r in rows]
        PE = [r["PE_eV"] for r in rows]
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 3.5))
        ax2.plot(t, E)
        ax2.set_xlabel("time (ps)")
        ax2.set_ylabel("Total E (eV)")
        ax2.grid(True, alpha=0.3)
        fig2.savefig(
            os.path.join(outdir, "energy_total.png"), dpi=300, bbox_inches="tight"
        )
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 3.5))
        ax3.plot(t, KE)
        ax3.set_xlabel("time (ps)")
        ax3.set_ylabel("KE (eV)")
        ax3.grid(True, alpha=0.3)
        fig3.savefig(
            os.path.join(outdir, "energy_kinetic.png"), dpi=300, bbox_inches="tight"
        )
        fig4, ax4 = plt.subplots(1, 1, figsize=(8, 3.5))
        ax4.plot(t, PE)
        ax4.set_xlabel("time (ps)")
        ax4.set_ylabel("PE (eV)")
        ax4.grid(True, alpha=0.3)
        fig4.savefig(
            os.path.join(outdir, "energy_potential.png"), dpi=300, bbox_inches="tight"
        )
    except Exception:
        pass

    logging.getLogger(__name__).info(
        f"NVE完成: 步数={steps}, 采样间隔={sample_every}, 输出: {outdir}"
    )
