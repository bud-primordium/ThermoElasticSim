#!/usr/bin/env python3
"""
零温弹性常数基准工作流（反哺自 examples/zero_temp_al_benchmark.py）

目标：
1. 将示例脚本中经过验证的计算流程沉淀为可复用 API
2. 统一材料常量来源，减少硬编码
3. 尽可能保持输出与绘图一致（可视化与 CSV 导出可选）

主要接口：
1. calculate_c11_c12_traditional(cell, potential, relaxer, mat)
2. calculate_c44_lammps_shear(cell, potential, relaxer, mat)

注意：
本模块不负责日志目录管理，调用方可按需配置日志与输出目录。
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from thermoelasticsim.core import CrystallineStructureBuilder
from thermoelasticsim.elastic.deformation_method.zero_temp import (
    ShearDeformationMethod,
    StructureRelaxer,
)
from thermoelasticsim.elastic.materials import (
    ALUMINUM_FCC,
    MaterialParameters,
)
from thermoelasticsim.potentials.eam import EAMAl1Potential
from thermoelasticsim.utils.utils import EV_TO_GPA
from thermoelasticsim.visualization.elastic.response_plotter import ResponsePlotter

logger = logging.getLogger(__name__)


def _setup_logging(output_dir: str | None = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # 控制台 handler：若不存在则添加，存在则调到期望级别
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    else:
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                h.setLevel(level)
    # 文件 handler：总是追加一个新的 run.log（如提供输出目录）
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            fh = logging.FileHandler(
                os.path.join(output_dir, "run.log"), mode="w", encoding="utf-8"
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception:
            logger.warning("无法创建日志文件处理器，继续仅输出到控制台。")


@dataclass(frozen=True)
class BenchmarkConfig:
    """基准参数配置"""

    supercell_size: tuple[int, int, int] = (3, 3, 3)
    # 形变法参数（C11/C12）
    delta: float = 0.003
    num_steps: int = 1  # 生产模式
    # 剪切法应变点（C44）
    shear_strains: tuple[float, ...] = (
        -0.004,
        -0.003,
        -0.002,
        -0.0015,
        -0.001,
        -0.0005,
        0.0,
        0.0005,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.004,
    )
    # 优化器
    optimizer_type: str = "L-BFGS"
    optimizer_params: dict[str, Any] = None  # type: ignore
    # 精度模式：用于极小应变（1e-6量级）下的线性拟合
    precision_mode: bool = False
    # 精度模式下的C11/C12应变点（线性极小幅，提升SNR到1e-5级）
    small_linear_strains: tuple[float, ...] = (
        -2e-5,
        -1e-5,
        0.0,
        1e-5,
        2e-5,
    )
    # 基态残余应力目标（GPa）与最大尝试次数（用于基态强化弛豫）
    base_stress_tol_gpa: float = 1e-5
    base_relax_max_passes: int = 3
    # 精度模式下的剪切应变点（仍保持在1e-4~1e-3量级以确保数值可测）
    small_shear_strains: tuple[float, ...] = (
        -0.0015,
        -0.001,
        -0.0005,
        0.0,
        0.0005,
        0.001,
        0.0015,
    )

    def build_relaxer(self) -> StructureRelaxer:
        """构建结构弛豫器"""
        # 默认或用户参数
        params = (
            {"ftol": 1e-7, "gtol": 1e-8, "maxiter": 5000, "maxls": 100}
            if self.optimizer_params is None
            else self.optimizer_params
        )
        # 精度模式：更严格的阈值
        if self.precision_mode:
            # 折中：在速度与精度间取中值（此前更严导致耗时过长）
            params = {
                **params,
                "ftol": 1e-15,
                "gtol": 1e-16,
                "maxiter": max(60000, params.get("maxiter", 0)),
            }

        return StructureRelaxer(
            optimizer_type=self.optimizer_type,
            optimizer_params=params,
            supercell_dims=self.supercell_size,
        )


def _generate_uniaxial_joint(
    base_cell,
    potential,
    relaxer: StructureRelaxer,
    strain_points: list[float] | np.ndarray,
    axis: int = 0,
    do_internal_relax: bool = True,
) -> tuple[list[dict], list[dict]]:
    """单次xx单轴应变，联合提取C11(σxx)与C12(σyy)两组数据。"""
    c11_rows: list[dict] = []
    c12_rows: list[dict] = []

    base = base_cell.copy()
    logger.info("制备无应力基态（优先等比例晶格弛豫）...")
    ok = relaxer.uniform_lattice_relax(base, potential)
    if not ok:
        logger.warning("等比例晶格弛豫未收敛，回退到完全弛豫（变胞+位置）")
        relaxer.full_relax(base, potential)

    base_stress = base.calculate_stress_tensor(potential) * EV_TO_GPA
    logger.info(
        "基态应力(GPa): [%.6f, %.6f, %.6f; %.6f, %.6f, %.6f; %.6f, %.6f, %.6f]",
        base_stress[0, 0],
        base_stress[0, 1],
        base_stress[0, 2],
        base_stress[1, 0],
        base_stress[1, 1],
        base_stress[1, 2],
        base_stress[2, 0],
        base_stress[2, 1],
        base_stress[2, 2],
    )

    for e in strain_points:
        if abs(e) < 1e-15:
            stress = base_stress
            converged = True
        else:
            F = np.eye(3)
            F[axis, axis] += e
            cell_e = base.copy()
            cell_e.apply_deformation(F)
            if do_internal_relax:
                converged = relaxer.internal_relax(cell_e, potential)
            else:
                converged = True
            stress = cell_e.calculate_stress_tensor(potential) * EV_TO_GPA
        logger.info(
            "单轴应变 εxx=%.6e → σxx=%.6f GPa, σyy=%.6f GPa（收敛=%s）",
            e,
            float(stress[0, 0]),
            float(stress[1, 1]),
            bool(converged),
        )

        # C11: σxx vs εxx
        c11_rows.append(
            {
                "applied_strain": float(e),
                "measured_stress_GPa": float(stress[0, 0]),
                "optimization_converged": bool(converged),
                "is_base": abs(e) < 1e-15,
            }
        )
        # C12: σyy vs εxx
        c12_rows.append(
            {
                "applied_strain": float(e),
                "measured_stress_GPa": float(stress[1, 1]),
                "optimization_converged": bool(converged),
                "is_base": abs(e) < 1e-15,
            }
        )

    return c11_rows, c12_rows


def calculate_c11_c12_traditional(
    cell,
    potential,
    relaxer: StructureRelaxer,
    material_params: MaterialParameters,
    strain_points: list[float] | np.ndarray | None = None,
    do_internal_relax: bool = True,
) -> dict:
    """使用单轴应变法计算 C11/C12，测试点与示例保持一致。"""
    # 若未提供，则采用示例风格：对称多点，含0点（与示例一致）
    if strain_points is None:
        strain_points = [
            -0.003,
            -0.002,
            -0.001,
            -0.0005,
            0.0,
            0.0005,
            0.001,
            0.002,
            0.003,
        ]

    # 一次xx单轴应变，同时获取C11与C12
    c11_data, c12_data = _generate_uniaxial_joint(
        base_cell=cell,
        potential=potential,
        relaxer=relaxer,
        strain_points=strain_points,
        axis=0,
        do_internal_relax=do_internal_relax,
    )

    # 仅用收敛点拟合斜率
    def _fit(data):
        arr = [
            (d["applied_strain"], d["measured_stress_GPa"])
            for d in data
            if d["optimization_converged"]
        ]
        if len(arr) < 2:
            return 0.0, 0.0
        x = np.array([a for a, _ in arr])
        y = np.array([b for _, b in arr])
        coeffs = np.polyfit(x, y, 1)
        ypred = np.polyval(coeffs, x)
        ss_res = np.sum((y - ypred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
        return float(coeffs[0]), float(r2)

    C11, r2_c11 = _fit(c11_data)
    C12, r2_c12 = _fit(c12_data)
    r2_score = float((r2_c11 + r2_c12) / 2)

    lit_C11 = material_params.literature_elastic_constants["C11"]
    lit_C12 = material_params.literature_elastic_constants["C12"]

    return {
        "C11": C11,
        "C12": C12,
        "C11_error_percent": abs(C11 - lit_C11) / lit_C11 * 100,
        "C12_error_percent": abs(C12 - lit_C12) / lit_C12 * 100,
        "r2_score": r2_score,
        "literature_C11": lit_C11,
        "literature_C12": lit_C12,
        "method": "uniaxial_traditional",
        "c11_data": c11_data,
        "c12_data": c12_data,
    }


def calculate_c44_lammps_shear(
    cell,
    potential,
    relaxer: StructureRelaxer,
    material_params: MaterialParameters,
    strain_points: np.ndarray | list[float],
) -> dict:
    """使用LAMMPS风格剪切方法计算C44/C55/C66。"""
    shear_method = ShearDeformationMethod()
    results = shear_method.calculate_c44_response(
        cell, potential, np.array(strain_points, dtype=float), relaxer
    )

    summary = results["summary"]
    C44_calculated = float(summary["C44"])  # 平均C44
    lit_C44 = material_params.literature_elastic_constants["C44"]

    return {
        "C44": C44_calculated,
        "C44_error_percent": abs(C44_calculated - lit_C44) / lit_C44 * 100,
        "average_r2_score": float(summary["average_r2_score"]),
        "converged_ratio": float(summary["converged_ratio"]),
        "literature_C44": lit_C44,
        "method": "LAMMPS_shear",
        "detailed_results": results["detailed_results"],
    }


def _fit_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """线性拟合 y = a x + b，返回 a 与 R²。"""
    coeffs = np.polyfit(x, y, 1)
    ypred = np.polyval(coeffs, x)
    ss_res = np.sum((y - ypred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
    return float(coeffs[0]), float(r2)


def calculate_c11_c12_robust(
    cell,
    potential,
    relaxer: StructureRelaxer,
    material_params: MaterialParameters,
    strain_points: list[float] | np.ndarray | None = None,
    do_internal_relax: bool = True,
) -> dict:
    """
    稳健法计算 C11/C12：正交应变(δ,-δ,0) + 等压应变(η,η,η)。

    - 正交：σxx = (C11 - C12)·δ, σyy = -(C11 - C12)·δ
    - 等压：σxx = (C11 + 2C12)·η = 3K·η

    通过两次一维拟合解出 C11、C12，通常比单轴法对势细节更鲁棒。
    """
    # 默认点集（含0点，双边对称）
    if strain_points is None:
        strain_points = [
            -0.003,
            -0.002,
            -0.001,
            -0.0005,
            0.0,
            0.0005,
            0.001,
            0.002,
            0.003,
        ]

    # 基态制备
    base = cell.copy()
    ok = relaxer.uniform_lattice_relax(base, potential)
    if not ok:
        relaxer.full_relax(base, potential)

    # --- 正交应变：F = diag(1+δ, 1-δ, 1)
    ortho_rows_xx: list[dict] = []
    ortho_rows_yy: list[dict] = []
    for d in strain_points:
        if abs(d) < 1e-15:
            stress = base.calculate_stress_tensor(potential) * EV_TO_GPA
            converged = True
        else:
            F = np.eye(3)
            F[0, 0] += d
            F[1, 1] -= d
            cell_d = base.copy()
            cell_d.apply_deformation(F)
            if do_internal_relax:
                converged = relaxer.internal_relax(cell_d, potential)
            else:
                converged = True
            stress = cell_d.calculate_stress_tensor(potential) * EV_TO_GPA

        ortho_rows_xx.append(
            {
                "applied_strain": float(d),
                "measured_stress_GPa": float(stress[0, 0]),
                "optimization_converged": bool(converged),
                "is_base": abs(d) < 1e-15,
            }
        )
        ortho_rows_yy.append(
            {
                "applied_strain": float(d),
                "measured_stress_GPa": float(stress[1, 1]),
                "optimization_converged": bool(converged),
                "is_base": abs(d) < 1e-15,
            }
        )

    # 仅用收敛点拟合 (σxx vs δ) 得到 Δ = C11 - C12
    arr_xx = [
        (r["applied_strain"], r["measured_stress_GPa"])
        for r in ortho_rows_xx
        if r["optimization_converged"]
    ]
    if len(arr_xx) < 2:
        delta_c = 0.0
        r2_ortho = 0.0
    else:
        x = np.array([a for a, _ in arr_xx])
        y = np.array([b for _, b in arr_xx])
        delta_c, r2_ortho = _fit_slope(x, y)

    # --- 等压应变：F = diag(1+η, 1+η, 1+η)
    hydro_rows_xx: list[dict] = []
    for eta in strain_points:
        if abs(eta) < 1e-15:
            stress = base.calculate_stress_tensor(potential) * EV_TO_GPA
            converged = True
        else:
            F = np.eye(3) * (1.0 + eta)
            cell_h = base.copy()
            cell_h.apply_deformation(F)
            if do_internal_relax:
                converged = relaxer.internal_relax(cell_h, potential)
            else:
                converged = True
            stress = cell_h.calculate_stress_tensor(potential) * EV_TO_GPA

        hydro_rows_xx.append(
            {
                "applied_strain": float(eta),
                "measured_stress_GPa": float(stress[0, 0]),
                "optimization_converged": bool(converged),
                "is_base": abs(eta) < 1e-15,
            }
        )

    arr_hx = [
        (r["applied_strain"], r["measured_stress_GPa"])
        for r in hydro_rows_xx
        if r["optimization_converged"]
    ]
    if len(arr_hx) < 2:
        slope_h = 0.0
        r2_hydro = 0.0
    else:
        xh = np.array([a for a, _ in arr_hx])
        yh = np.array([b for _, b in arr_hx])
        slope_h, r2_hydro = _fit_slope(xh, yh)  # slope_h = C11 + 2 C12 = 3K

    # 解出 C11, C12
    K = slope_h / 3.0
    C11 = K + (2.0 / 3.0) * delta_c
    C12 = K - (1.0 / 3.0) * delta_c

    lit_C11 = material_params.literature_elastic_constants["C11"]
    lit_C12 = material_params.literature_elastic_constants["C12"]

    return {
        "C11": float(C11),
        "C12": float(C12),
        "C11_error_percent": abs(C11 - lit_C11) / lit_C11 * 100,
        "C12_error_percent": abs(C12 - lit_C12) / lit_C12 * 100,
        "r2_score": float(0.5 * (r2_ortho + r2_hydro)),
        "literature_C11": lit_C11,
        "literature_C12": lit_C12,
        "method": "orthorhombic_plus_hydrostatic",
        "c11_data": ortho_rows_xx,  # 用于绘图：视为C11曲线
        "c12_data": ortho_rows_yy,  # 用于绘图：视为C12曲线（符号相反）
        "details": {
            "delta_c": float(delta_c),
            "K": float(K),
            "slope_hydro": float(slope_h),
            "r2_ortho": float(r2_ortho),
            "r2_hydro": float(r2_hydro),
        },
    }


def calculate_c11_c12_biaxial_orthorhombic(
    cell,
    potential,
    relaxer: StructureRelaxer,
    material_params: MaterialParameters,
    strain_points: list[float] | np.ndarray | None = None,
    do_internal_relax: bool = True,
) -> dict:
    """
    通过两种无体积拟合获取 C11/C12：

    - 正交: F = diag(1+δ, 1-δ, 1) → 斜率 Δ = C11 - C12
    - 双轴: F = diag(1+ε, 1+ε, 1) → 斜率 Σ = C11 + C12

    最终：C11 = (Σ+Δ)/2, C12 = (Σ-Δ)/2
    避免直接使用等压路径对体积敏感的数值漂移。
    """
    if strain_points is None:
        strain_points = [-0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003]

    base = cell.copy()
    ok = relaxer.uniform_lattice_relax(base, potential)
    if not ok:
        relaxer.full_relax(base, potential)

    # 正交
    ortho_rows_xx: list[dict] = []
    for d in strain_points:
        if abs(d) < 1e-15:
            stress = base.calculate_stress_tensor(potential) * EV_TO_GPA
            converged = True
        else:
            F = np.diag([1.0 + d, 1.0 - d, 1.0])
            c = base.copy()
            c.apply_deformation(F)
            if do_internal_relax:
                converged = relaxer.internal_relax(c, potential)
            else:
                converged = True
            stress = c.calculate_stress_tensor(potential) * EV_TO_GPA
        ortho_rows_xx.append(
            {
                "applied_strain": float(d),
                "measured_stress_GPa": float(stress[0, 0]),
                "optimization_converged": bool(converged),
                "is_base": abs(d) < 1e-15,
            }
        )

    arr_o = [
        (r["applied_strain"], r["measured_stress_GPa"])
        for r in ortho_rows_xx
        if r["optimization_converged"]
    ]
    # 若有效点过少，回退到无内部弛豫以补足点数
    if len(arr_o) < 3 and do_internal_relax:
        ortho_rows_xx.clear()
        for d in strain_points:
            if abs(d) < 1e-15:
                stress = base.calculate_stress_tensor(potential) * EV_TO_GPA
                converged = True
            else:
                F = np.diag([1.0 + d, 1.0 - d, 1.0])
                c = base.copy()
                c.apply_deformation(F)
                converged = True
                stress = c.calculate_stress_tensor(potential) * EV_TO_GPA
            ortho_rows_xx.append(
                {
                    "applied_strain": float(d),
                    "measured_stress_GPa": float(stress[0, 0]),
                    "optimization_converged": True,
                    "is_base": abs(d) < 1e-15,
                }
            )
        arr_o = [
            (r["applied_strain"], r["measured_stress_GPa"])
            for r in ortho_rows_xx
            if r["optimization_converged"]
        ]
    if len(arr_o) < 2:
        delta_c, r2_o = 0.0, 0.0
    else:
        x = np.array([a for a, _ in arr_o])
        y = np.array([b for _, b in arr_o])
        delta_c, r2_o = _fit_slope(x, y)

    # 双轴
    biax_rows_xx: list[dict] = []
    for e in strain_points:
        if abs(e) < 1e-15:
            stress = base.calculate_stress_tensor(potential) * EV_TO_GPA
            converged = True
        else:
            F = np.diag([1.0 + e, 1.0 + e, 1.0])
            c = base.copy()
            c.apply_deformation(F)
            if do_internal_relax:
                converged = relaxer.internal_relax(c, potential)
            else:
                converged = True
            stress = c.calculate_stress_tensor(potential) * EV_TO_GPA
        biax_rows_xx.append(
            {
                "applied_strain": float(e),
                "measured_stress_GPa": float(stress[0, 0]),
                "optimization_converged": bool(converged),
                "is_base": abs(e) < 1e-15,
            }
        )

    arr_b = [
        (r["applied_strain"], r["measured_stress_GPa"])
        for r in biax_rows_xx
        if r["optimization_converged"]
    ]
    if len(arr_b) < 3 and do_internal_relax:
        biax_rows_xx.clear()
        for e in strain_points:
            if abs(e) < 1e-15:
                stress = base.calculate_stress_tensor(potential) * EV_TO_GPA
                converged = True
            else:
                F = np.diag([1.0 + e, 1.0 + e, 1.0])
                c = base.copy()
                c.apply_deformation(F)
                converged = True
                stress = c.calculate_stress_tensor(potential) * EV_TO_GPA
            biax_rows_xx.append(
                {
                    "applied_strain": float(e),
                    "measured_stress_GPa": float(stress[0, 0]),
                    "optimization_converged": True,
                    "is_base": abs(e) < 1e-15,
                }
            )
        arr_b = [
            (r["applied_strain"], r["measured_stress_GPa"])
            for r in biax_rows_xx
            if r["optimization_converged"]
        ]
    if len(arr_b) < 2:
        sigma_c, r2_b = 0.0, 0.0
    else:
        xb = np.array([a for a, _ in arr_b])
        yb = np.array([b for _, b in arr_b])
        sigma_c, r2_b = _fit_slope(xb, yb)

    C11 = 0.5 * (sigma_c + delta_c)
    C12 = 0.5 * (sigma_c - delta_c)

    lit_C11 = material_params.literature_elastic_constants["C11"]
    lit_C12 = material_params.literature_elastic_constants["C12"]

    return {
        "C11": float(C11),
        "C12": float(C12),
        "C11_error_percent": abs(C11 - lit_C11) / lit_C11 * 100,
        "C12_error_percent": abs(C12 - lit_C12) / lit_C12 * 100,
        "r2_score": float(0.5 * (r2_o + r2_b)),
        "literature_C11": lit_C11,
        "literature_C12": lit_C12,
        "method": "biaxial_plus_orthorhombic",
        "c11_data": ortho_rows_xx,
        "c12_data": biax_rows_xx,
        "details": {
            "delta_c": float(delta_c),
            "sigma_c": float(sigma_c),
            "r2_orthorhombic": float(r2_o),
            "r2_biaxial": float(r2_b),
        },
    }


def run_zero_temp_benchmark(
    material_params: MaterialParameters,
    potential,
    supercell_size: tuple[int, int, int] = (3, 3, 3),
    output_dir: str | None = None,
    save_json: bool = True,
    precision: bool = False,
    log_level: int | None = None,
) -> dict:
    """运行通用零温弹性基准（结构自适应）。

    - 根据 `material_params.structure` 自动生成晶体结构（当前支持 fcc、diamond）。
    - 其它流程保持与铝基准一致：弛豫、C11/C12 单轴、C44 剪切、绘图与CSV/JSON。
    """
    mat = material_params
    nx, ny, nz = supercell_size

    # 结构与势
    builder = CrystallineStructureBuilder()
    if mat.structure == "fcc":
        cell = builder.create_fcc(mat.symbol, mat.lattice_constant, supercell_size)
    elif mat.structure == "diamond":
        cell = builder.create_diamond(mat.symbol, mat.lattice_constant, supercell_size)
    else:
        raise NotImplementedError(f"暂不支持的晶体结构: {mat.structure}")
    total_atoms = cell.num_atoms

    _setup_logging(
        output_dir, level=(log_level if log_level is not None else logging.INFO)
    )
    logger.info(
        f"开始{mat.symbol}弹性常数基准：超胞={supercell_size}（原子数={total_atoms}）"
    )

    # 弛豫器
    cfg = BenchmarkConfig(supercell_size=supercell_size, precision_mode=precision)
    relaxer = cfg.build_relaxer()

    # 不做额外基态强化弛豫：遵循默认流程，避免多余耗时与日志噪声

    # 计算C11/C12：对Cu默认采用稳健法；其他材料维持传统法（可按需切换）
    libname = getattr(getattr(potential, "cpp_interface", object()), "_lib_name", "")
    # 尺寸-截断检查（避免最小镜像违规）
    try:
        cutoff = float(getattr(potential, "cutoff", 0.0) or 0.0)
        lengths = cell.get_box_lengths()
        min_half = float(np.min(lengths) / 2.0)
        if cutoff > 0.0 and min_half <= cutoff:
            logger.warning(
                "超胞过小: 最小半盒长 %.3f Å <= 截断半径 %.3f Å。建议增大尺寸以减少体积相关误差。",
                min_half,
                cutoff,
            )
    except Exception:
        pass
    if libname == "eam_cu1":
        # 为避免图意义混淆，Cu 统一采用传统单轴法（斜率即为 C11/C12）
        if precision:
            c11_c12 = calculate_c11_c12_traditional(
                cell,
                potential,
                relaxer,
                mat,
                strain_points=list(cfg.small_linear_strains),
                do_internal_relax=True,
            )
        else:
            c11_c12 = calculate_c11_c12_traditional(
                cell, potential, relaxer, mat, strain_points=None
            )
    else:
        if precision:
            c11_c12 = calculate_c11_c12_traditional(
                cell,
                potential,
                relaxer,
                mat,
                strain_points=list(cfg.small_linear_strains),
                do_internal_relax=True,
            )
        else:
            c11_c12 = calculate_c11_c12_traditional(
                cell, potential, relaxer, mat, strain_points=None
            )

    # 计算C44
    c44_strains = cfg.small_shear_strains if precision else cfg.shear_strains
    c44 = calculate_c44_lammps_shear(
        cell, potential, relaxer, mat, strain_points=c44_strains
    )

    # 汇总
    C11, C12, C44 = c11_c12["C11"], c11_c12["C12"], c44["C44"]
    bulk_modulus_calc = (C11 + 2 * C12) / 3
    shear_modulus_calc = C44
    young_modulus_calc = (
        9
        * bulk_modulus_calc
        * shear_modulus_calc
        / (3 * bulk_modulus_calc + shear_modulus_calc)
    )
    poisson_ratio_calc = (3 * bulk_modulus_calc - 2 * shear_modulus_calc) / (
        6 * bulk_modulus_calc + 2 * shear_modulus_calc
    )

    results: dict[str, Any] = {
        "supercell_size": supercell_size,
        "total_atoms": total_atoms,
        "material": mat.name,
        "elastic_constants": {"C11": C11, "C12": C12, "C44": C44},
        "elastic_moduli": {
            "bulk_modulus": bulk_modulus_calc,
            "shear_modulus": shear_modulus_calc,
            "young_modulus": young_modulus_calc,
            "poisson_ratio": poisson_ratio_calc,
        },
        "errors": {
            "C11_error_percent": c11_c12["C11_error_percent"],
            "C12_error_percent": c11_c12["C12_error_percent"],
            "C44_error_percent": c44["C44_error_percent"],
            "average_error_percent": (
                c11_c12["C11_error_percent"]
                + c11_c12["C12_error_percent"]
                + c44["C44_error_percent"]
            )
            / 3,
        },
        "quality_metrics": {
            "c11_c12_r2": c11_c12["r2_score"],
            "c44_r2": c44["average_r2_score"],
            "c44_converged_ratio": c44["converged_ratio"],
        },
        "methods_used": {
            "c11_c12_method": c11_c12["method"],
            "c44_method": c44["method"],
        },
    }

    # 可选：图与JSON输出
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # C44剪切响应图（传入材料文献值以绘制正确参考线）
        plotter = ResponsePlotter(
            literature_values={
                "C11": mat.literature_elastic_constants.get("C11", 0.0),
                "C12": mat.literature_elastic_constants.get("C12", 0.0),
                "C44": mat.literature_elastic_constants.get("C44", 0.0),
                "C55": mat.literature_elastic_constants.get(
                    "C55", mat.literature_elastic_constants.get("C44", 0.0)
                ),
                "C66": mat.literature_elastic_constants.get(
                    "C66", mat.literature_elastic_constants.get("C44", 0.0)
                ),
            }
        )
        shear_png = os.path.join(
            output_dir,
            f"c44_c55_c66_shear_response_{nx}x{ny}x{nz}.png",
        )
        plotter.plot_shear_response(
            detailed_results=c44["detailed_results"],
            supercell_size=supercell_size,
            output_path=shear_png,
        )
        results.setdefault("artifacts", {})["c44_plot"] = os.path.basename(shear_png)

        # C44 剪切 CSV 汇总（与示例风格一致）
        c44_rows: list[dict[str, Any]] = []
        for dr in c44["detailed_results"]:
            name = dr.get("name", "")
            # 提取标签：yz/xz/xy（去掉中文等非xyz字符）
            raw = name.split("(")[0] if "(" in name else name
            label = "".join(ch for ch in raw if ch in "xyz") or raw
            for s, t, ok in zip(
                dr["strains"], dr["stresses"], dr["converged_states"], strict=False
            ):
                c44_rows.append(
                    {
                        "calculation_method": "C44_shear",
                        "applied_strain_direction": label,
                        "applied_strain": float(s),
                        "measured_stress_GPa": float(t),
                        "optimization_converged": bool(ok),
                        "is_reference_state": abs(s) < 1e-15,
                    }
                )
        c44_csv = os.path.join(
            output_dir,
            f"c44_shear_analysis_{nx}x{ny}x{nz}.csv",
        )
        pd.DataFrame(c44_rows).to_csv(c44_csv, index=False)
        results["artifacts"]["c44_csv"] = os.path.basename(c44_csv)

        # C11/C12 联合图
        c11c12_png = os.path.join(
            output_dir,
            f"c11_c12_combined_response_{nx}x{ny}x{nz}.png",
        )
        # 传统单轴法下，斜率即为常数，直接绘制原始散点与拟合线
        plotter.plot_c11_c12_combined_response(
            c11_data=c11_c12["c11_data"],
            c12_data=c11_c12["c12_data"],
            supercell_size=supercell_size,
            output_path=c11c12_png,
        )
        results["artifacts"]["c11c12_plot"] = os.path.basename(c11c12_png)

        # C11/C12 联合CSV
        c11c12_rows: list[dict[str, Any]] = []
        for row in c11_c12["c11_data"]:
            c11c12_rows.append(
                {
                    "calculation_method": "C11_uniaxial",
                    "applied_strain_direction": "xx",
                    "applied_strain": float(row["applied_strain"]),
                    "measured_stress_GPa": float(row["measured_stress_GPa"]),
                    "optimization_converged": bool(row["optimization_converged"]),
                    "is_reference_state": bool(row.get("is_base", False)),
                }
            )
        for row in c11_c12["c12_data"]:
            c11c12_rows.append(
                {
                    "calculation_method": "C12_cross",
                    "applied_strain_direction": "yy",
                    "applied_strain": float(row["applied_strain"]),
                    "measured_stress_GPa": float(row["measured_stress_GPa"]),
                    "optimization_converged": bool(row["optimization_converged"]),
                    "is_reference_state": bool(row.get("is_base", False)),
                }
            )
        c11c12_csv = os.path.join(output_dir, f"c11_c12_combined_{nx}x{ny}x{nz}.csv")
        pd.DataFrame(c11c12_rows).to_csv(c11c12_csv, index=False)
        results["artifacts"]["c11c12_csv"] = os.path.basename(c11c12_csv)

        # 弹性常数汇总CSV（含文献与误差）
        summary_csv = os.path.join(output_dir, f"elastic_constants_{nx}x{ny}x{nz}.csv")
        lit_C11 = mat.literature_elastic_constants["C11"]
        lit_C12 = mat.literature_elastic_constants["C12"]
        lit_C44 = mat.literature_elastic_constants["C44"]
        row = {
            "size": f"{nx}x{ny}x{nz}",
            "C11_GPa": C11,
            "C12_GPa": C12,
            "C44_GPa": C44,
            "lit_C11_GPa": lit_C11,
            "lit_C12_GPa": lit_C12,
            "lit_C44_GPa": lit_C44,
            "err_C11_percent": abs(C11 - lit_C11) / lit_C11 * 100,
            "err_C12_percent": abs(C12 - lit_C12) / lit_C12 * 100,
            "err_C44_percent": abs(C44 - lit_C44) / lit_C44 * 100,
            "bulk_modulus_GPa": bulk_modulus_calc,
            "shear_modulus_GPa": shear_modulus_calc,
            "young_modulus_GPa": young_modulus_calc,
            "poisson_ratio": poisson_ratio_calc,
            "r2_c11c12": results["quality_metrics"]["c11_c12_r2"],
            "r2_c44": results["quality_metrics"]["c44_r2"],
            "c44_convergence_ratio": results["quality_metrics"]["c44_converged_ratio"],
        }
        pd.DataFrame([row]).to_csv(summary_csv, index=False)
        results["artifacts"]["elastic_constants_csv"] = os.path.basename(summary_csv)

        # JSON
        if save_json:
            json_path = os.path.join(output_dir, "benchmark_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            results.setdefault("artifacts", {})["json"] = os.path.basename(json_path)

    return results


## 统一入口为 run_zero_temp_benchmark；冗余包装已移除。


def run_size_sweep(
    sizes: list[tuple[int, int, int]] | None = None,
    output_root: str | None = None,
    material_params: MaterialParameters | None = None,
    potential_factory: Callable[[], Any] | None = None,
    precision: bool = False,
) -> list[dict]:
    """依次运行 2x2x2、3x3x3、4x4x4 超胞的有限尺寸效应基准。

    - 默认材料为铝（ALUMINUM_FCC），默认势为 EAMAl1Potential。
    - 可通过传入 material_params 与 potential_factory 进行泛化。
    """
    if sizes is None:
        sizes = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
    logger = logging.getLogger(__name__)

    mat = material_params or ALUMINUM_FCC
    pot_factory = potential_factory or EAMAl1Potential

    results = []
    for s in sizes:
        outdir = None
        if output_root:
            outdir = os.path.join(output_root, f"{s[0]}x{s[1]}x{s[2]}")
        t0 = time.time()
        res = run_zero_temp_benchmark(
            material_params=mat,
            potential=pot_factory(),
            supercell_size=s,
            output_dir=outdir,
            save_json=True,
            precision=precision,
        )
        dt = time.time() - t0
        res["duration_sec"] = dt
        results.append(res)
        # 输出尺寸对比信息（与旧版类似）
        C11 = res["elastic_constants"]["C11"]
        C12 = res["elastic_constants"]["C12"]
        C44 = res["elastic_constants"]["C44"]
        e11 = res["errors"]["C11_error_percent"]
        e12 = res["errors"]["C12_error_percent"]
        e44 = res["errors"]["C44_error_percent"]
        r2_c = res["quality_metrics"].get("c11_c12_r2", 0.0)
        r2_s = res["quality_metrics"].get("c44_r2", 0.0)
        nx, ny, nz = res["supercell_size"]
        logger.info(
            f"尺寸 {nx}×{ny}×{nz}: C11={C11:.2f} GPa (误差 {e11:.2f}%), "
            f"C12={C12:.2f} GPa (误差 {e12:.2f}%), C44={C44:.2f} GPa (误差 {e44:.2f}%), "
            f"R²(C11/C12)={r2_c:.3f}, R²(C44)={r2_s:.3f}, 用时 {dt:.2f}s"
        )
    # 合并尺寸结果CSV
    if output_root:
        combined_path = os.path.join(output_root, "combined_elastic_data.csv")
        rows: list[dict[str, Any]] = []
        for r in results:
            nx, ny, nz = r["supercell_size"]
            C11 = r["elastic_constants"]["C11"]
            C12 = r["elastic_constants"]["C12"]
            C44 = r["elastic_constants"]["C44"]
            rows.append(
                {
                    "size": f"{nx}x{ny}x{nz}",
                    "C11_GPa": C11,
                    "C12_GPa": C12,
                    "C44_GPa": C44,
                    "lit_C11_GPa": mat.literature_elastic_constants["C11"],
                    "lit_C12_GPa": mat.literature_elastic_constants["C12"],
                    "lit_C44_GPa": mat.literature_elastic_constants["C44"],
                    "err_C11_percent": r["errors"]["C11_error_percent"],
                    "err_C12_percent": r["errors"]["C12_error_percent"],
                    "err_C44_percent": r["errors"]["C44_error_percent"],
                    "bulk_modulus_GPa": r["elastic_moduli"]["bulk_modulus"],
                    "shear_modulus_GPa": r["elastic_moduli"]["shear_modulus"],
                    "young_modulus_GPa": r["elastic_moduli"]["young_modulus"],
                    "poisson_ratio": r["elastic_moduli"]["poisson_ratio"],
                    "r2_c11c12": r["quality_metrics"]["c11_c12_r2"],
                    "r2_c44": r["quality_metrics"]["c44_r2"],
                    "c44_convergence_ratio": r["quality_metrics"][
                        "c44_converged_ratio"
                    ],
                }
            )
        pd.DataFrame(rows).to_csv(combined_path, index=False)
    return results
