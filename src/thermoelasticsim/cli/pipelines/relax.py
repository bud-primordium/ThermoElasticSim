"""零温弛豫教学场景流水线"""

from __future__ import annotations

import json
import logging
import os

from ...elastic import StructureRelaxer
from ...utils.utils import EV_TO_GPA
from .common import build_cell, get_material_by_spec, make_potential


def run_relax_pipeline(
    cfg, outdir: str, material_symbol: str, potential_kind: str
) -> None:
    """运行零温结构弛豫，并导出快照与 E–V/E–s 曲线。

    Notes
    -----
    支持三种模式：uniform（等比例缩放）、full（变胞+位置优化）、auto（默认）。
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

    mode = str(cfg.get("relax.mode", "auto")).lower()
    relaxer = StructureRelaxer(
        optimizer_type="L-BFGS",
        optimizer_params={"ftol": 1e-7, "gtol": 1e-6, "maxiter": 5000},
        supercell_dims=sc,
    )

    logger = logging.getLogger(__name__)
    e0 = pot.calculate_energy(cell)
    v0 = float(cell.volume)
    s0 = cell.calculate_stress_tensor(pot) * EV_TO_GPA

    converged = False
    used_mode = mode
    if mode == "uniform":
        converged = relaxer.uniform_lattice_relax(cell, pot)
    elif mode == "full":
        converged = relaxer.full_relax(cell, pot)
    else:
        used_mode = "uniform"
        converged = relaxer.uniform_lattice_relax(cell, pot)
        if not converged:
            used_mode = "full"
            converged = relaxer.full_relax(cell, pot)

    e1 = pot.calculate_energy(cell)
    v1 = float(cell.volume)
    s1 = cell.calculate_stress_tensor(pot) * EV_TO_GPA

    # 生成 E–V / E–s 曲线
    try:
        import matplotlib.pyplot as plt
        import numpy as _np

        final_lattice = cell.lattice_vectors.copy()
        final_frac = cell.get_fractional_coordinates()
        scales = _np.linspace(0.98, 1.02, 21)
        vols: list[float] = []
        enes: list[float] = []
        for s in scales:
            scaled = final_lattice * float(s)
            cell.set_lattice_vectors(scaled)
            cell.set_fractional_coordinates(final_frac)
            enes.append(float(pot.calculate_energy(cell)))
            vols.append(float(cell.volume))
        cell.set_lattice_vectors(final_lattice)
        cell.set_fractional_coordinates(final_frac)
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.0))
        ax.plot(vols, enes, marker="o", ms=3)
        ax.set_xlabel("体积 (Å³)")
        ax.set_ylabel("能量 (eV)")
        ax.set_title("E–V 曲线 (最终晶格附近)")
        ax.grid(True, alpha=0.3)
        fig.savefig(
            os.path.join(outdir, "energy_vs_volume.png"), dpi=300, bbox_inches="tight"
        )
        fig2, ax2 = plt.subplots(1, 1, figsize=(6.4, 4.0))
        ax2.plot(scales, enes, marker="o", ms=3)
        ax2.set_xlabel("缩放因子 s")
        ax2.set_ylabel("能量 (eV)")
        ax2.set_title("E–s 曲线")
        ax2.grid(True, alpha=0.3)
        fig2.savefig(
            os.path.join(outdir, "energy_vs_scale.png"), dpi=300, bbox_inches="tight"
        )
    except Exception:
        pass

    snapshot = {
        "used_mode": used_mode,
        "converged": bool(converged),
        "initial": {"energy_eV": float(e0), "volume_A3": v0, "stress_GPa": s0.tolist()},
        "final": {
            "energy_eV": float(e1),
            "volume_A3": v1,
            "stress_GPa": s1.tolist(),
            "lattice_vectors": cell.lattice_vectors.tolist(),
            "num_atoms": cell.num_atoms,
            "symbols": [a.symbol for a in cell.atoms],
            "positions": [a.position.tolist() for a in cell.atoms],
        },
        "material": {
            "name": mat.name,
            "symbol": mat.symbol,
            "structure": mat.structure,
        },
        "supercell": list(sc),
        "potential": getattr(
            getattr(pot, "cpp_interface", object()), "_lib_name", type(pot).__name__
        ),
    }
    with open(os.path.join(outdir, "relax_results.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    logger.info(
        f"弛豫完成: 模式={used_mode}, 收敛={converged}, E0={e0:.6f}→E1={e1:.6f} eV, V0={v0:.3f}→V1={v1:.3f} Å³"
    )
