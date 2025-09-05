#!/usr/bin/env python3
"""
零温铜弹性常数基准测试 - 新组件版本

使用重构后的统一组件进行Cu弹性常数计算，展示通用API在FCC材料上的复用。
与铝示例保持输出/绘图风格一致，但使用铜的文献值与势函数。

功能特点:
- 复用 CrystallineStructureBuilder 生成FCC结构
- 复用通用 run_zero_temp_benchmark 工作流
- 使用 EAMCu1Potential 与 COPPER_FCC 文献常数
- 提供尺寸扫描与汇总输出（2×2×2, 3×3×3, 4×4×4）

.. moduleauthor:: Gilbert Young
.. created:: 2025-08-24
.. version:: 4.0.0
"""

import logging
import os
import sys
from datetime import datetime

from thermoelasticsim.core.config import ConfigManager
from thermoelasticsim.elastic import COPPER_FCC, run_zero_temp_benchmark
from thermoelasticsim.potentials.eam import EAMCu1Potential


def setup_logging(test_name: str = "cu_benchmark", run_dir: str | None = None) -> str:
    """设置日志系统并创建运行目录"""
    base_logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_dir is None:
        run_dir = os.path.join(base_logs_dir, f"{test_name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

    log_filename = f"{test_name}_{timestamp}.log"
    log_filepath = os.path.join(run_dir, log_filename)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return run_dir


def run_copper_benchmark_local(
    supercell_size: tuple[int, int, int] = (3, 3, 3), output_dir: str | None = None
) -> dict:
    """运行完整的铜弹性常数基准测试"""
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("铜弹性常数基准测试 - 新组件版本")
    nx, ny, nz = supercell_size
    logger.info(f"超胞尺寸: {nx}×{ny}×{nz}")
    logger.info("=" * 80)

    results = run_zero_temp_benchmark(
        material_params=COPPER_FCC,
        potential=EAMCu1Potential(),
        supercell_size=supercell_size,
        output_dir=output_dir,
        save_json=True,
        precision=True,
    )

    # 输出最终结果（与铝示例风格一致）
    logger.info("\n" + "=" * 60)
    logger.info("最终结果汇总")
    logger.info("=" * 60)
    mat = COPPER_FCC
    logger.info("弹性常数 (GPa):")
    logger.info(f"  实际原子数: {results['total_atoms']}")
    logger.info(
        f"  C11: {results['elastic_constants']['C11']:.2f} (文献: {mat.literature_elastic_constants['C11']:.2f})"
    )
    logger.info(
        f"  C12: {results['elastic_constants']['C12']:.2f} (文献: {mat.literature_elastic_constants['C12']:.2f})"
    )
    logger.info(
        f"  C44: {results['elastic_constants']['C44']:.2f} (文献: {mat.literature_elastic_constants['C44']:.2f})"
    )
    logger.info("\n弹性模量对比:")
    logger.info(
        f"  体积模量: {results['elastic_moduli']['bulk_modulus']:.2f} GPa (理论: {mat.bulk_modulus:.2f} GPa)"
    )
    logger.info(
        f"  剪切模量: {results['elastic_moduli']['shear_modulus']:.2f} GPa (理论: {mat.shear_modulus:.2f} GPa)"
    )
    logger.info(
        f"  杨氏模量: {results['elastic_moduli']['young_modulus']:.2f} GPa (理论: {mat.young_modulus:.2f} GPa)"
    )
    logger.info(
        f"  泊松比: {results['elastic_moduli']['poisson_ratio']:.3f} (理论: {mat.poisson_ratio:.3f})"
    )

    total_error = results["errors"]["average_error_percent"]
    logger.info("\n计算质量:")
    logger.info(f"  平均误差: {total_error:.2f}%")
    logger.info(f"  C11/C12拟合优度: {results['quality_metrics']['c11_c12_r2']:.6f}")
    logger.info(f"  C44拟合优度: {results['quality_metrics']['c44_r2']:.6f}")
    logger.info(f"  C44收敛率: {results['quality_metrics']['c44_converged_ratio']:.1%}")

    quality = (
        "优秀" if total_error < 5.0 else ("良好" if total_error < 10.0 else "需改进")
    )
    logger.info(f"  总体评估: {quality}")
    results.setdefault("quality_metrics", {})["overall_quality"] = quality
    return results


def main():
    cfg = ConfigManager()
    seed = cfg.set_global_seed()
    run_name = cfg.get("run.name", "cu_benchmark")
    run_dir = cfg.make_output_dir(run_name)
    cfg.snapshot(run_dir)
    run_dir = setup_logging(run_name, run_dir)
    logging.getLogger(__name__).info(f"随机数种子: {seed}")
    logger = logging.getLogger(__name__)
    try:
        sizes = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
        # 走统一入口：对size逐一用run_zero_temp_benchmark
        results_list = []
        for s in sizes:
            outdir = os.path.join(run_dir, f"{s[0]}x{s[1]}x{s[2]}")
            os.makedirs(outdir, exist_ok=True)
            res = run_zero_temp_benchmark(
                material_params=COPPER_FCC,
                potential=EAMCu1Potential(),
                supercell_size=s,
                output_dir=outdir,
                save_json=True,
                precision=True,
            )
            results_list.append(res)

        # 保存合并结果
        import json

        results_file = os.path.join(run_dir, "size_sweep_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)

        logger.info("\n尺寸汇总对比：")
        for res in results_list:
            nx, ny, nz = res["supercell_size"]
            C11 = res["elastic_constants"]["C11"]
            C12 = res["elastic_constants"]["C12"]
            C44 = res["elastic_constants"]["C44"]
            e11 = res["errors"]["C11_error_percent"]
            e12 = res["errors"]["C12_error_percent"]
            e44 = res["errors"]["C44_error_percent"]
            r2c = res["quality_metrics"].get("c11_c12_r2", 0.0)
            r2s = res["quality_metrics"].get("c44_r2", 0.0)
            dur = res.get("duration_sec", 0.0)
            logger.info(
                f"尺寸 {nx}×{ny}×{nz}: C11={C11:.2f} GPa (误差 {e11:.2f}%), "
                f"C12={C12:.2f} GPa (误差 {e12:.2f}%), C44={C44:.2f} GPa (误差 {e44:.2f}%), "
                f"R²(C11/C12)={r2c:.3f}, R²(C44)={r2s:.3f}, 用时 {dur:.2f}s"
            )

        logger.info("\n尺寸扫描完成！")
        logger.info(f"结果保存到: {run_dir}")
        logger.info(f"详细日志: {os.path.join(run_dir, 'cu_benchmark_*.log')}")
        logger.info(f"合并结果: {results_file}")

        return results_list
    except Exception as e:
        logger.error(f"基准测试失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
