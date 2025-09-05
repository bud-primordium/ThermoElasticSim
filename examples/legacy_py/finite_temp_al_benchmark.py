#!/usr/bin/env python3
"""
完整有限温弹性常数计算 - 基于 MTK-NPT 预平衡 + NHC 采样

功能概览：
- NPT 预平衡：MTK 可逆积分（含压力/体积/温度记录与压力演化图）
- NVT 生产：NHC 精控温度，采样应力以拟合 C11/C12/C44
- 统计输出：应力-应变图（误差棒）、文献对比、JSON 结果存档

压力与符号约定：
- 应力张量 σ 采用拉伸为正（tension-positive）；压力 P = -tr(σ)/3
- 目标外压通常设为 0 GPa；评估以时间平均为准（非瞬时值）

使用建议（压力控制精度）：
- 小体系和短时间下瞬时压力涨落较大（常达数百 MPa），不必追求 1e-4 GPa
- 建议以数 ps 的时间平均（或移动平均）接近目标压强为准
- 提高精度的手段：更大超胞（≥256 原子）、更长平均窗、更温和的 pdamp、更小 dt

Created: 2025-08-20
"""

import contextlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# 尝试加载统一的绘图风格（可选）
with contextlib.suppress(Exception):
    from thermoelasticsim.utils.plot_config import plt  # type: ignore


# 将 src 加入路径
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from thermoelasticsim.core.config import ConfigManager
from thermoelasticsim.core.structure import Atom, Cell
from thermoelasticsim.elastic.deformation_method.zero_temp import StructureRelaxer
from thermoelasticsim.elastic.mechanics import StressCalculator
from thermoelasticsim.md.propagators import NoseHooverChainPropagator
from thermoelasticsim.md.schemes import create_mtk_npt_scheme
from thermoelasticsim.potentials.eam import EAMAl1Potential
from thermoelasticsim.utils.utils import EV_TO_GPA, KB_IN_EV

# ============================
# 通用工具与初始化
# ============================


def setup_logging(tag: str, run_dir: str | None = None) -> tuple[str, str]:
    base_logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_dir is None:
        run_dir = os.path.join(base_logs_dir, f"{tag}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

    log_filepath = os.path.join(run_dir, f"{tag}_{timestamp}.log")

    # 清理根 logger 旧 handler
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_filepath, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return log_filepath, run_dir


def create_aluminum_fcc_4x4x4() -> Cell:
    """构建 4x4x4 FCC Al（256 原子）"""
    a = 4.045
    nx = ny = nz = 4
    lattice = np.array(
        [[a * nx, 0, 0], [0, a * ny, 0], [0, 0, a * nz]], dtype=np.float64
    )
    atoms: list[Atom] = []
    atom_id = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pos_list = [
                    [i * a, j * a, k * a],
                    [i * a + a / 2, j * a + a / 2, k * a],
                    [i * a + a / 2, j * a, k * a + a / 2],
                    [i * a, j * a + a / 2, k * a + a / 2],
                ]
                for p in pos_list:
                    atoms.append(
                        Atom(
                            id=atom_id,
                            symbol="Al",
                            mass_amu=26.9815,
                            position=np.array(p),
                        )
                    )
                    atom_id += 1
    return Cell(lattice, atoms)


def initialize_maxwell(cell: Cell, T: float) -> float:
    for atom in cell.atoms:
        sigma = np.sqrt(KB_IN_EV * T / atom.mass)
        atom.velocity = np.random.normal(0.0, sigma, 3)
    cell.remove_com_motion()
    return cell.calculate_temperature()


# ============================
# 基态制备与预热
# ============================


def zero_temp_relax(cell: Cell, pot: EAMAl1Potential, logger: logging.Logger) -> bool:
    relaxer = StructureRelaxer(
        optimizer_type="L-BFGS",
        optimizer_params={"ftol": 1e-7, "gtol": 1e-6, "maxiter": 2000},
        supercell_dims=(4, 4, 4),
    )
    e0 = pot.calculate_energy(cell)
    v0 = cell.get_volume()
    logger.info(f"零温初始: E={e0:.6f} eV, V={v0:.3f} Å³")
    t0 = time.time()
    ok = relaxer.uniform_lattice_relax(cell, pot)
    dt = time.time() - t0
    if ok:
        e1 = pot.calculate_energy(cell)
        v1 = cell.get_volume()
        logger.info(f"零温收敛: E={e1:.6f} eV, V={v1:.3f} Å³, 用时{dt:.1f}s")
        base_stress = StressCalculator().compute_stress(cell, pot) * EV_TO_GPA
        logger.info(f"基态应力范数: {np.linalg.norm(base_stress):.4f} GPa")
    else:
        logger.error("零温基态制备失败")
    return ok


def run_langevin_preheat(
    cell: Cell,
    pot: EAMAl1Potential,
    T: float,
    logger: logging.Logger,
    *,
    dt: float = 0.5,
    steps: int = 10000,
    friction: float = 3.0,
) -> dict[str, list[float]]:
    from thermoelasticsim.md.schemes import LangevinNVTScheme

    gamma = friction  # ps^-1 强耦合快速升温

    initialize_maxwell(cell, 20.0)
    scheme = LangevinNVTScheme(target_temperature=T, friction=gamma)
    hist = {"t_ps": [], "T": [], "E": []}
    logger.info(
        f"Langevin 预热: γ={gamma} ps^-1, {steps}步 ({steps * dt / 1000:.1f} ps)"
    )
    for s in range(steps):
        scheme.step(cell, pot, dt)
        if s % 200 == 0:
            t_ps = s * dt / 1000.0
            ke = sum(
                0.5 * atom.mass * np.dot(atom.velocity, atom.velocity)
                for atom in cell.atoms
            )
            pe = pot.calculate_energy(cell)
            hist["t_ps"].append(t_ps)
            hist["T"].append(cell.calculate_temperature())
            hist["E"].append(ke + pe)
            if s % 2000 == 0:
                logger.info(f"预热 {s:5d}: T={hist['T'][-1]:.1f} K")
    return hist


# ============================
# MTK-NPT 预平衡（含压力演化图）
# ============================


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    c = np.convolve(x, np.ones(w), mode="valid") / w
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, c])


def run_mtk_npt(
    cell: Cell,
    pot: EAMAl1Potential,
    T: float,
    P_target_GPa: float,
    logger: logging.Logger,
    run_dir: str,
    *,
    dt: float = 0.15,
    tdamp: float = 50.0,
    pdamp: float = 150.0,
    steps: int = 20000,
    sample_every: int = 50,
    ma_window_ps: float = 0.5,
) -> dict[str, Any]:
    """运行 MTK-NPT 并记录压力/体积/温度，输出压力演化图。

    说明：
    - 默认为 256 原子、dt=0.15 fs、pdamp=150 fs，压力更平滑（比 30 fs 更稳）
    - steps=20000 即 3 ps；可按需延长至 20–50 ps 获得更小统计误差
    """
    logger.info(
        f"MTK-NPT: T={T} K, P={P_target_GPa} GPa, dt={dt} fs, tdamp={tdamp} fs, pdamp={pdamp} fs, steps={steps}"
    )

    scheme = create_mtk_npt_scheme(
        target_temperature=T,
        target_pressure=P_target_GPa,
        tdamp=tdamp,
        pdamp=pdamp,
        tchain=3,
        pchain=3,
    )

    # 记录
    t_ps: list[float] = []
    temps: list[float] = []
    pressures_gpa: list[float] = []
    volumes: list[float] = []
    stress_calc = StressCalculator()

    for s in range(steps):
        scheme.step(cell, pot, dt)
        if (s % sample_every) == 0:
            t_ps.append(s * dt / 1000.0)
            temps.append(cell.calculate_temperature())
            # 使用应力迹计算压力（不做“近零裁剪”）
            sigma = stress_calc.calculate_total_stress(cell, pot)
            P_gpa = (-np.trace(sigma) / 3.0) / 6.2415e-3
            pressures_gpa.append(P_gpa)
            volumes.append(cell.volume)
        if (s % (2000)) == 0:
            logger.info(
                f"NPT {s:6d}: T={temps[-1]:.1f}K, P={pressures_gpa[-1]:+.3f} GPa, V={volumes[-1]:.2f} Å³"
            )

    # 压力演化图
    t = np.array(t_ps)
    P = np.array(pressures_gpa)
    # T_hist = np.array(temps)  # 暂未绘制温度历史图
    # V_hist = np.array(volumes)  # 暂未绘制体积历史图

    ma_win = max(1, int(ma_window_ps * 1000.0 / dt / sample_every))
    P_ma = moving_average(P, ma_win)

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
        y=P_target_GPa,
        color="#D55E00",
        linestyle="--",
        label=f"目标压力 {P_target_GPa} GPa",
    )
    ax.set_xlabel("时间 (ps)")
    ax.set_ylabel("压力 (GPa)")
    ax.set_title("MTK-NPT 压力演化")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(run_dir, "npt_pressure_evolution.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"NPT 压力演化图已保存: {fig_path}")

    # 返回末态与统计
    return {
        "t_ps": t_ps,
        "T": temps,
        "P_GPa": pressures_gpa,
        "V": volumes,
        "pressure_plot": fig_path,
        "final_volume": cell.volume,
        "final_temperature": cell.calculate_temperature(),
        "avg_pressure_GPa": float(np.nanmean(P_ma[np.isfinite(P_ma)]))
        if len(P_ma) > 0
        else float(np.mean(P)),
        "std_pressure_GPa": float(np.nanstd(P_ma[np.isfinite(P_ma)]))
        if len(P_ma) > 0
        else float(np.std(P)),
    }


# ============================
# NHC 生产采样（应力统计）
# ============================


def apply_strain(cell: Cell, mode: str, magnitude: float) -> Cell:
    eps = np.zeros((3, 3))
    if mode == "xx":
        eps[0, 0] = magnitude
    elif mode == "yy":
        eps[1, 1] = magnitude
    elif mode == "zz":
        eps[2, 2] = magnitude
    elif mode == "yz":
        eps[1, 2] = eps[2, 1] = magnitude / 2.0  # 工程剪切的一半
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


def nhc_sample_stress(
    cell: Cell,
    pot: EAMAl1Potential,
    T: float,
    *,
    dt: float = 0.5,
    tdamp: float = 50.0,
    pre_steps: int = 10000,
    prod_steps: int = 20000,
    sample_every: int = 50,
) -> dict[str, Any]:
    nhc = NoseHooverChainPropagator(
        target_temperature=T, tdamp=tdamp, tchain=3, tloop=1
    )
    stress_calc = StressCalculator()

    # 预平衡
    for s in range(pre_steps):
        nhc.propagate(cell, dt)
        if (s + 1) % 2000 == 0:
            pass

    # 生产采样
    stresses: list[np.ndarray] = []
    temps: list[float] = []
    for s in range(prod_steps):
        nhc.propagate(cell, dt)
        if (s + 1) % sample_every == 0:
            sigma = stress_calc.compute_stress(cell, pot) * EV_TO_GPA  # GPa
            stresses.append(sigma)
            temps.append(cell.calculate_temperature())

    if not stresses:
        return {"success": False}

    S = np.array(stresses)
    T_hist = np.array(temps)
    return {
        "success": True,
        "avg_sigma": S.mean(axis=0),
        "std_sigma": S.std(axis=0),
        "avg_T": float(T_hist.mean()),
        "std_T": float(T_hist.std()),
        "n_samples": int(S.shape[0]),
    }


def fit_elastic_constant(
    strains: np.ndarray, stresses: np.ndarray
) -> tuple[float, np.ndarray]:
    coeffs = np.polyfit(strains, stresses, 1)
    return float(coeffs[0]), coeffs


# ============================
# 主流程
# ============================


def main():
    # 配置与种子
    cfg = ConfigManager()
    cfg.set_global_seed()

    # 输出目录按配置生成
    run_name = cfg.get("run.name", "finite_temp_elastic_mtk_complete")
    run_dir = cfg.make_output_dir(run_name)
    cfg.snapshot(run_dir)

    log_path, run_dir = setup_logging(run_name, run_dir)
    logger = logging.getLogger(__name__)
    logger.info("=" * 88)
    logger.info("有限温弹性常数 - MTK-NPT 完整流程 (C11/C12/C44)")
    logger.info(f"日志: {log_path}")
    logger.info(f"输出: {run_dir}")
    logger.info("=" * 88)

    target_T = float(cfg.get("md.temperature", 300.0))  # K
    target_P = float(cfg.get("md.pressure", 0.0))  # GPa (常压)

    # 1) 构建体系与零温基态
    cell = create_aluminum_fcc_4x4x4()
    pot = EAMAl1Potential()
    if not zero_temp_relax(cell, pot, logger):
        return 1

    # base_cell = cell.copy()  # 备份原始晶胞（暂未使用）

    # 2) 预热 + MTK-NPT 预平衡（带压力演化图）
    pre_dt = float(cfg.get("finite_temp.preheat.dt", 0.5))
    pre_steps = int(cfg.get("finite_temp.preheat.steps", 10000))
    pre_gamma = float(
        cfg.get(
            "finite_temp.preheat.friction",
            cfg.get("thermostats.langevin.friction", 3.0),
        )
    )
    _ = run_langevin_preheat(
        cell, pot, target_T, logger, dt=pre_dt, steps=pre_steps, friction=pre_gamma
    )
    npt_stats = run_mtk_npt(
        cell,
        pot,
        target_T,
        target_P,
        logger,
        run_dir,
        dt=float(cfg.get("finite_temp.npt.dt", 0.15)),
        tdamp=float(
            cfg.get(
                "finite_temp.npt.tdamp", cfg.get("thermostats.nose_hoover.tdamp", 50.0)
            )
        ),
        pdamp=float(
            cfg.get("finite_temp.npt.pdamp", cfg.get("barostats.mtk.pdamp", 150.0))
        ),
        steps=int(cfg.get("finite_temp.npt.steps", 20000)),
        sample_every=int(cfg.get("finite_temp.npt.sample_every", 50)),
        ma_window_ps=float(cfg.get("finite_temp.npt.ma_window_ps", 0.5)),
    )
    logger.info(
        f"NPT 结果: <P>={npt_stats['avg_pressure_GPa']:+.3f}±{npt_stats['std_pressure_GPa']:.3f} GPa, V={npt_stats['final_volume']:.2f} Å³"
    )

    # 记录 NPT 末态残余应力（作为零点偏置进行校正）
    sigma_zero = StressCalculator().compute_stress(cell, pot) * EV_TO_GPA  # GPa
    logger.info(
        f"NPT 末态残余应力 (GPa): diag={np.diag(sigma_zero)}; P={-np.trace(sigma_zero) / 3.0:+.3f}"
    )

    eq_cell = cell.copy()

    # 3) 应变方案与采样（默认不单独加“0%”点；使用 NPT 末态残余应力作偏置）
    # 按需求：不包含0%，上限0.5%，采用8点：±0.20%、±0.30%、±0.40%、±0.50%
    strain_set = {
        "C11": {
            "mode": "xx",
            "magnitudes": np.array(
                [-0.005, -0.004, -0.003, -0.002, 0.002, 0.003, 0.004, 0.005]
            ),
        },
        "C12": {
            "mode": "xx",
            "magnitudes": np.array(
                [-0.005, -0.004, -0.003, -0.002, 0.002, 0.003, 0.004, 0.005]
            ),
        },
        # C44 剪切应变的 magnitude 以工程剪切 γ 表示，apply_strain 内部已除以2写入应变张量
        "C44": {
            "mode": "yz",
            "magnitudes": np.array(
                [-0.005, -0.004, -0.003, -0.002, 0.002, 0.003, 0.004, 0.005]
            ),
        },
    }

    # 逐项采样
    results: dict[str, Any] = {
        "meta": {
            "system": "Al FCC 4x4x4 (256 atoms)",
            "T": target_T,
            "P_target_GPa": target_P,
            "npt_pressure_plot": npt_stats["pressure_plot"],
            "npt_pressure_mean_GPa": npt_stats["avg_pressure_GPa"],
            "npt_pressure_std_GPa": npt_stats["std_pressure_GPa"],
        },
        "C11": {},
        "C12": {},
        "C44": {},
    }

    # 采样与拟合工具
    def do_series(name: str, mode: str, mags: np.ndarray) -> dict[str, Any]:
        xs: list[float] = []
        ys: list[float] = []
        es: list[float] = []
        samples: list[dict[str, Any]] = []
        for m in mags:
            dcell = apply_strain(eq_cell, mode, float(m))
            stat = nhc_sample_stress(
                dcell,
                pot,
                target_T,
                dt=float(cfg.get("finite_temp.nhc.dt", 0.5)),
                tdamp=float(
                    cfg.get(
                        "finite_temp.nhc.tdamp",
                        cfg.get("thermostats.nose_hoover.tdamp", 50.0),
                    )
                ),
                pre_steps=int(cfg.get("finite_temp.nhc.pre_steps", 6000)),
                prod_steps=int(cfg.get("finite_temp.nhc.prod_steps", 20000)),
                sample_every=int(cfg.get("finite_temp.nhc.sample_every", 50)),
            )
            if not stat.get("success", False):
                logger.error(f"{name} {m:+.2%} 采样失败")
                continue
            # 取相关应力分量（减去零点偏置）
            sig_avg = stat["avg_sigma"]  # GPa
            sig_std = stat["std_sigma"]

            if name in ("C11", "C12"):
                # 线弹性：εxx→σxx 对应 C11；εxx→σyy 对应 C12
                comp = (0, 0) if name == "C11" else (1, 1)
            else:  # C44: εyz→τyz
                comp = (1, 2)

            # 零点偏置来自 NPT 末态应力
            y = float(sig_avg[comp] - sigma_zero[comp])
            e = float(sig_std[comp])

            xs.append(float(m))
            ys.append(y)
            es.append(e)
            samples.append(
                {
                    "strain": float(m),
                    "sigma": sig_avg.tolist(),
                    "sigma_std": sig_std.tolist(),
                    "temp_avg": stat["avg_T"],
                    "temp_std": stat["std_T"],
                }
            )

        if len(xs) >= 2:
            C, coeffs = fit_elastic_constant(np.array(xs), np.array(ys))
        else:
            C, coeffs = float("nan"), np.array([np.nan, np.nan])

        return {
            "points": samples,
            "strains": xs,
            "stresses": ys,
            "errors": es,
            "value": C,
            "coeffs": coeffs.tolist(),
        }

    res_C11 = do_series(
        "C11", strain_set["C11"]["mode"], strain_set["C11"]["magnitudes"]
    )
    res_C12 = do_series(
        "C12", strain_set["C12"]["mode"], strain_set["C12"]["magnitudes"]
    )
    res_C44 = do_series(
        "C44", strain_set["C44"]["mode"], strain_set["C44"]["magnitudes"]
    )

    results["C11"] = res_C11
    results["C12"] = res_C12
    results["C44"] = res_C44

    # 4) 可视化：每个常数一张图
    def plot_series(name: str, data: dict[str, Any]) -> str:
        xs = np.array(data.get("strains", []))
        ys = np.array(data.get("stresses", []))
        es = np.array(data.get("errors", []))
        coeffs = np.array(data.get("coeffs", [0, 0]))
        Cval = data.get("value", float("nan"))

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
        plt.tight_layout()
        path = os.path.join(run_dir, f"{name}_fit.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    path_C11 = plot_series("C11", res_C11)
    path_C12 = plot_series("C12", res_C12)
    path_C44 = plot_series("C44", res_C44)

    results["C11"]["plot"] = path_C11
    results["C12"]["plot"] = path_C12
    results["C44"]["plot"] = path_C44

    # 5) 输出 JSON
    json_path = os.path.join(run_dir, "finite_temp_elastic_mtk_complete.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON 结果已保存: {json_path}")

    # 6) 控制台总结
    def fmt(x: Any) -> str:
        return (
            "nan"
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
            else f"{x:.1f}"
        )

    logger.info("-" * 60)
    logger.info(f"C11={fmt(res_C11.get('value'))} GPa; 图: {path_C11}")
    logger.info(f"C12={fmt(res_C12.get('value'))} GPa; 图: {path_C12}")
    logger.info(f"C44={fmt(res_C44.get('value'))} GPa; 图: {path_C44}")
    logger.info("-" * 60)
    logger.info("完成：建议扩展更长 NPT/NVT 时间与更大超胞以提升精度")
    return 0


if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except Exception as e:
        print(f"运行失败: {e}")
        raise
