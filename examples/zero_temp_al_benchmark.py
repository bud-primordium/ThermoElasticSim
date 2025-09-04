#!/usr/bin/env python3
"""
零温铝弹性常数基准测试 - 新组件版本

使用重构后的统一组件进行Al弹性常数计算，展示新的API使用方法。
相比原版本，此版本更简洁且利用了可复用的组件架构。

功能特点:
- 使用CrystallineStructureBuilder统一生成FCC结构
- 使用MaterialParameters配置和验证材料参数
- 使用传统形变矩阵方法计算C11/C12
- 使用ShearDeformationMethod(LAMMPS风格)计算C44
- 统一的结果输出和误差分析

.. moduleauthor:: Gilbert Young
.. created:: 2025-08-24
.. version:: 4.0.0
"""

import logging
import os
import sys
from datetime import datetime

# 核心组件
from thermoelasticsim.elastic import (
    ALUMINUM_FCC,
    BenchmarkConfig,
    run_size_sweep,
    run_zero_temp_benchmark,
)
from thermoelasticsim.potentials.eam import EAMAl1Potential


def setup_logging(test_name: str = "al_benchmark") -> str:
    """
    设置日志系统并创建运行目录

    Returns
    -------
    str
        运行目录路径
    """
    # 创建独立的运行目录
    base_logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_logs_dir, f"{test_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    log_filename = f"{test_name}_{timestamp}.log"
    log_filepath = os.path.join(run_dir, log_filename)

    # 清除现有handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()
    # 根logger设为DEBUG，控制台handler控制输出为INFO，文件保留DEBUG
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 文件handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return run_dir


def run_aluminum_benchmark_local(
    supercell_size: tuple[int, int, int] = (3, 3, 3), output_dir: str | None = None
) -> dict:
    """
    运行完整的铝弹性常数基准测试

    Parameters
    ----------
    supercell_size : tuple[int, int, int], optional
        超胞尺寸，默认(3,3,3)

    Returns
    -------
    dict
        包含所有计算结果的字典
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("铝弹性常数基准测试 - 新组件版本")
    nx, ny, nz = supercell_size
    logger.info(f"超胞尺寸: {nx}×{ny}×{nz}")
    logger.info("=" * 80)

    # 使用库内统一工作流运行，保持示例脚本轻量化
    cfg = BenchmarkConfig(supercell_size=supercell_size)
    results = run_zero_temp_benchmark(
        material_params=ALUMINUM_FCC,
        potential=EAMAl1Potential(),
        supercell_size=cfg.supercell_size,
        output_dir=output_dir,
        save_json=True,
        precision=False,
    )

    # 输出最终结果（与原示例风格相近）
    logger.info("\n" + "=" * 60)
    logger.info("最终结果汇总")
    logger.info("=" * 60)
    mat = ALUMINUM_FCC
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

    # 质量评估
    if total_error < 5.0:
        quality = "优秀"
    elif total_error < 10.0:
        quality = "良好"
    else:
        quality = "需改进"

    logger.info(f"  总体评估: {quality}")

    # 返回完整结果
    results["quality_metrics"]["overall_quality"] = quality
    return results


def main():
    """主函数：运行基准测试"""
    # 设置日志
    run_dir = setup_logging("al_benchmark")
    logger = logging.getLogger(__name__)

    try:
        # 尺寸扫描：2x2x2, 3x3x3, 4x4x4（与旧版保持一致）
        sizes = [(2, 2, 2), (3, 3, 3), (4, 4, 4)]
        results_list = run_size_sweep(sizes=sizes, output_root=run_dir)

        # 保存合并结果到文件
        import json

        results_file = os.path.join(run_dir, "size_sweep_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=2, ensure_ascii=False)

        # 屏幕输出最终汇总对比（与旧版风格一致）
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
        logger.info(f"详细日志: {os.path.join(run_dir, 'al_benchmark_*.log')}")
        logger.info(f"合并结果: {results_file}")

        return results_list

    except Exception as e:
        logger.error(f"基准测试失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
