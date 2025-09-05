"""CLI 场景通用工具

提供材料/势函数解析、晶体构建、初速分配、常用绘图等工具函数，
供各个场景流水线复用（relax/zero_temp/finite_temp/nve/nvt/npt）。

Notes
-----
这些函数面向 CLI 级别的拼装逻辑，避免在核心库中引入场景耦合。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ...core import CrystallineStructureBuilder
from ...elastic import ALUMINUM_FCC, CARBON_DIAMOND, COPPER_FCC, GOLD_FCC
from ...utils.utils import KB_IN_EV


def make_potential(kind: str):
    """按字符串创建常用势函数实例。

    Parameters
    ----------
    kind : str
        势函数标识；支持 'EAM_Al1'、'EAM_Cu1'、'Tersoff_C1988'（大小写不敏感）。

    Returns
    -------
    object
        势函数对象实例。
    """
    k = kind.strip().lower()
    if k in ("eam_al1", "al_eam", "eam:al"):
        from ...potentials.eam import EAMAl1Potential

        return EAMAl1Potential()
    if k in ("eam_cu1", "cu_eam", "eam:cu"):
        from ...potentials.eam import EAMCu1Potential

        return EAMCu1Potential()
    if k in ("tersoff_c1988", "tersoff:c1988", "tersoff_c"):
        from ...potentials.tersoff import TersoffC1988Potential

        return TersoffC1988Potential(delta=0.0)
    raise ValueError(f"未知势函数: {kind}")


def get_material_by_spec(symbol: str, structure: str | None = None):
    """按材料符号与可选同素异形体返回预定义材料参数。

    当前支持:
    - Al: fcc
    - Cu: fcc
    - C : diamond

    Parameters
    ----------
    symbol : str
        元素符号，如 'Al'、'Cu'、'C'。
    structure : str, optional
        同素异形体，如 'fcc'、'diamond'。可为 None 使用默认。

    Returns
    -------
    MaterialParameters
    """
    s = (symbol or "").strip()
    st = (structure or "").strip().lower()
    if s == "Al":
        return ALUMINUM_FCC
    if s == "Cu":
        return COPPER_FCC
    if s == "Au":
        return GOLD_FCC
    if s == "C" and st in ("diamond", "", None):
        return CARBON_DIAMOND
    raise ValueError(f"未知材料/结构组合: symbol={symbol}, structure={structure}")


def build_cell(material, supercell: tuple[int, int, int]):
    """根据材料结构构建晶胞。"""
    builder = CrystallineStructureBuilder()
    if material.structure == "fcc":
        return builder.create_fcc(material.symbol, material.lattice_constant, supercell)
    if material.structure == "diamond":
        return builder.create_diamond(
            material.symbol, material.lattice_constant, supercell
        )
    raise NotImplementedError(f"暂不支持的结构: {material.structure}")


def assign_maxwell(cell, temperature: float) -> None:
    """按 Maxwell 分布为原子分配初速并去除整体平动。"""
    for atom in cell.atoms:
        sigma = np.sqrt(KB_IN_EV * temperature / atom.mass)
        atom.velocity = np.random.normal(0.0, sigma, 3)
    cell.remove_com_motion()


def apply_strain(cell, mode: str, magnitude: float):
    """施加简单应变（xx/yy/zz 或 xy/xz/yz），返回新晶胞副本。"""
    eps = np.zeros((3, 3))
    if mode == "xx":
        eps[0, 0] = magnitude
    elif mode == "yy":
        eps[1, 1] = magnitude
    elif mode == "zz":
        eps[2, 2] = magnitude
    elif mode == "yz":
        eps[1, 2] = eps[2, 1] = magnitude / 2.0
    elif mode == "xz":
        eps[0, 2] = eps[2, 0] = magnitude / 2.0
    elif mode == "xy":
        eps[0, 1] = eps[1, 0] = magnitude / 2.0
    else:
        raise ValueError(f"未知应变类型: {mode}")
    F = np.eye(3) + eps
    out = cell.copy()
    out.apply_deformation(F)
    return out


def plot_series(
    run_dir: str, name: str, xs: Iterable[float], ys, es, coeffs, Cval
) -> str:
    """绘制带误差棒的应力-应变拟合图。返回图路径。"""
    import os

    import matplotlib.pyplot as plt

    xs = np.array(list(xs))
    ys = np.array(ys)
    es = np.array(es)
    coeffs = np.array(coeffs)
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5))
    ax.errorbar(
        xs * 100.0, ys, yerr=es, fmt="o", capsize=4, color="#1f77b4", label="采样点"
    )
    if np.isfinite(Cval) and len(xs) >= 2:
        xf = np.linspace(xs.min(), xs.max(), 100)
        yf = np.polyval(coeffs, xf)
        ax.plot(
            xf * 100.0,
            yf,
            "--",
            color="#D55E00",
            label=f"线性拟合: {name}={Cval:.1f} GPa",
        )
    ax.grid(True, alpha=0.3)
    if name == "C44":
        ax.set_xlabel("剪切应变 γyz (%)")
        ax.set_ylabel("剪切应力 τyz (GPa)")
    else:
        ax.set_xlabel("应变 εxx (%)")
        ax.set_ylabel(f"应力 σ{'xx' if name == 'C11' else 'yy'} (GPa)")
    ax.set_title(f"{name} 有限温弹性常数（NPT零点偏置已校正）")
    ax.legend()
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{name}_fit.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path
