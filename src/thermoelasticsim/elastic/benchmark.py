#!/usr/bin/env python3
"""
零温弹性常数基准工作流（反哺自 examples/zero_temp_al_benchmark.py）

目标：
- 将示例脚本中经过验证的计算流程沉淀为可复用API
- 统一材料常量来源，减少硬编码
- 保持输出与绘图尽可能一致（通过可选的可视化与CSV导出）

主要接口：
- run_aluminum_benchmark(supercell_size) → dict
- calculate_c11_c12_traditional(cell, potential, relaxer, mat)
- calculate_c44_lammps_shear(cell, potential, relaxer, mat)

注意：
本模块不负责日志目录管理，调用方可按需配置日志与输出目录。
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from thermoelasticsim.core import CrystallineStructureBuilder
from thermoelasticsim.elastic.deformation_method.zero_temp import (
    ShearDeformationMethod,
    StructureRelaxer,
)
from thermoelasticsim.elastic.materials import ALUMINUM_FCC, MaterialParameters
from thermoelasticsim.potentials.eam import EAMAl1Potential
from thermoelasticsim.utils.utils import EV_TO_GPA
from thermoelasticsim.visualization.elastic.response_plotter import ResponsePlotter

logger = logging.getLogger(__name__)


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

    def build_relaxer(self) -> StructureRelaxer:
        """构建结构弛豫器"""
        return StructureRelaxer(
            optimizer_type=self.optimizer_type,
            optimizer_params=(
                {"ftol": 1e-7, "gtol": 1e-8, "maxiter": 5000, "maxls": 100}
                if self.optimizer_params is None
                else self.optimizer_params
            ),
            supercell_dims=self.supercell_size,
        )


def _generate_uniaxial_joint(
    base_cell,
    potential,
    relaxer: StructureRelaxer,
    strain_points: list[float] | np.ndarray,
    axis: int = 0,
) -> tuple[list[dict], list[dict]]:
    """单次xx单轴应变，联合提取C11(σxx)与C12(σyy)两组数据。"""
    c11_rows: list[dict] = []
    c12_rows: list[dict] = []

    base = base_cell.copy()
    ok = relaxer.uniform_lattice_relax(base, potential)
    if not ok:
        relaxer.full_relax(base, potential)

    base_stress = base.calculate_stress_tensor(potential) * EV_TO_GPA

    for e in strain_points:
        if abs(e) < 1e-15:
            stress = base_stress
            converged = True
        else:
            F = np.eye(3)
            F[axis, axis] += e
            cell_e = base.copy()
            cell_e.apply_deformation(F)
            converged = relaxer.internal_relax(cell_e, potential)
            stress = cell_e.calculate_stress_tensor(potential) * EV_TO_GPA

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


def run_aluminum_benchmark(
    supercell_size: tuple[int, int, int] = (3, 3, 3),
    output_dir: str | None = None,
    save_json: bool = True,
) -> dict:
    """运行完整的铝零温弹性基准（C11/C12/C44）。"""
    mat = ALUMINUM_FCC
    nx, ny, nz = supercell_size
    total_atoms = 4 * nx * ny * nz

    logger.info(f"开始铝弹性常数基准：超胞={supercell_size}（原子数={total_atoms}）")

    # 结构与势
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(mat.symbol, mat.lattice_constant, supercell_size)
    potential = EAMAl1Potential()

    # 弛豫器
    cfg = BenchmarkConfig(supercell_size=supercell_size)
    relaxer = cfg.build_relaxer()

    # 计算C11/C12
    c11_c12 = calculate_c11_c12_traditional(
        cell, potential, relaxer, mat, strain_points=None
    )

    # 计算C44
    c44 = calculate_c44_lammps_shear(
        cell, potential, relaxer, mat, strain_points=cfg.shear_strains
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

        # C44剪切响应图
        plotter = ResponsePlotter()
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


def run_size_sweep(
    sizes: list[tuple[int, int, int]] | None = None,
    output_root: str | None = None,
) -> list[dict]:
    """依次运行 2x2x2、3x3x3、4x4x4 超胞的有限尺寸效应基准。"""
    if sizes is None:
        sizes = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
    logger = logging.getLogger(__name__)
    results = []
    for s in sizes:
        outdir = None
        if output_root:
            outdir = os.path.join(output_root, f"{s[0]}x{s[1]}x{s[2]}")
        t0 = time.time()
        res = run_aluminum_benchmark(
            supercell_size=s, output_dir=outdir, save_json=True
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
        nx, ny, nz = res["supercell_size"]
        logger.info(
            f"尺寸 {nx}×{ny}×{nz}: C11={C11:.2f} GPa (误差 {e11:.2f}%), "
            f"C12={C12:.2f} GPa (误差 {e12:.2f}%), C44={C44:.2f} GPa (误差 {e44:.2f}%), 用时 {dt:.2f}s"
        )
    # 合并尺寸结果CSV
    if output_root:
        combined_path = os.path.join(output_root, "combined_elastic_data.csv")
        rows: list[dict[str, Any]] = []
        for r in results:
            nx, ny, nz = r["supercell_size"]
            mat = ALUMINUM_FCC
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
