#!/usr/bin/env python3
"""YAML 场景入口（CLI）

使用示例::

    python -m thermoelasticsim.cli.run -c examples/modern_yaml/zero_temp_elastic.yaml

说明
----
- 本入口只负责 YAML 解析与场景调度；具体实现见 ``pipelines/*`` 模块。
"""

from __future__ import annotations

import argparse
import logging
import os

from thermoelasticsim.core.config import ConfigManager
from thermoelasticsim.elastic.benchmark import (
    _setup_logging as _setup_benchmark_logging,
)

from .pipelines.finite_temp import run_finite_temp_pipeline
from .pipelines.npt import run_npt_pipeline
from .pipelines.nve import run_nve_pipeline
from .pipelines.nvt import run_nvt_pipeline
from .pipelines.relax import run_relax_pipeline
from .pipelines.zero_temp import run_zero_temp_pipeline


def main(argv: list[str] | None = None) -> int:
    """解析 YAML 并调度对应场景。"""
    ap = argparse.ArgumentParser(description="ThermoElasticSim: YAML 驱动运行入口")
    ap.add_argument("-c", "--config", required=True, help="YAML配置文件路径")
    args = ap.parse_args(argv)

    cfg = ConfigManager(files=[args.config])
    seed = cfg.set_global_seed()

    name = cfg.get("run.name", cfg.get("scenario", "run"))
    outdir = cfg.make_output_dir(name)
    _setup_benchmark_logging(outdir, level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info(f"随机数种子: {seed}")

    scenario = str(cfg.get("scenario", "zero_temp")).lower()
    material_symbol = str(cfg.get("material.symbol", cfg.get("material", "Al")))
    material_structure = cfg.get("material.structure", None)
    potential_kind = str(cfg.get("potential.type", cfg.get("potential", "EAM_Al1")))

    # 写入精简版有效配置（便于复查）
    effective = {
        "scenario": scenario,
        "run": {"name": cfg.get("run.name", "run")},
        "rng": {"global_seed": cfg.get("rng.global_seed", seed)},
        "material": {"symbol": material_symbol, "structure": material_structure},
        "potential": {"type": potential_kind},
    }
    if scenario in ("zero_temp", "zerotemp", "zt"):
        effective["elastic"] = {
            "sizes": cfg.get("elastic.sizes", cfg.get("supercell", None))
        }
        effective["precision"] = bool(cfg.get("precision", False))
    elif scenario in ("finite_temp", "finitetemp", "ft"):
        effective["supercell"] = cfg.get("supercell", [4, 4, 4])
        effective["md"] = {
            "temperature": cfg.get("md.temperature", 300.0),
            "pressure": cfg.get("md.pressure", 0.0),
        }
        effective["finite_temp"] = {
            "preheat": cfg.get("finite_temp.preheat", {}),
            "npt": cfg.get("finite_temp.npt", {}),
            "nhc": cfg.get("finite_temp.nhc", {}),
        }
    try:
        import yaml

        with open(
            os.path.join(outdir, "effective_config.yaml"), "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(effective, f, allow_unicode=True, sort_keys=True)
    except Exception:
        pass

    log.info(
        f"场景: {scenario} | 材料: {material_symbol} ({material_structure or 'default'}) | 势: {potential_kind}"
    )

    if scenario in ("relax", "relax_only"):
        run_relax_pipeline(cfg, outdir, material_symbol, potential_kind)
    elif scenario in ("zero_temp", "zerotemp", "zt"):
        results = run_zero_temp_pipeline(cfg, outdir, material_symbol, potential_kind)
        if isinstance(results, list):
            log.info("\n尺寸汇总对比：")
            for r in results:
                nx, ny, nz = r["supercell_size"]
                C11 = r["elastic_constants"]["C11"]
                C12 = r["elastic_constants"]["C12"]
                C44 = r["elastic_constants"]["C44"]
                e11 = r["errors"]["C11_error_percent"]
                e12 = r["errors"]["C12_error_percent"]
                e44 = r["errors"]["C44_error_percent"]
                r2c = r["quality_metrics"].get("c11_c12_r2", 0.0)
                r2s = r["quality_metrics"].get("c44_r2", 0.0)
                log.info(
                    f"尺寸 {nx}×{ny}×{nz}: C11={C11:.2f} GPa (误差 {e11:.2f}%), "
                    f"C12={C12:.2f} GPa (误差 {e12:.2f}%), C44={C44:.2f} GPa (误差 {e44:.2f}%), "
                    f"R²(C11/C12)={r2c:.3f}, R²(C44)={r2s:.3f}"
                )
    elif scenario in ("finite_temp", "finitetemp", "ft"):
        run_finite_temp_pipeline(cfg, outdir, material_symbol, potential_kind)
    elif scenario in ("nve",):
        run_nve_pipeline(cfg, outdir, material_symbol, potential_kind)
    elif scenario in ("nvt",):
        run_nvt_pipeline(cfg, outdir, material_symbol, potential_kind)
    elif scenario in ("npt",):
        run_npt_pipeline(cfg, outdir, material_symbol, potential_kind)
    else:
        raise ValueError(f"未知场景类型 scenario: {scenario}")

    log.info(f"完成。输出目录: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
