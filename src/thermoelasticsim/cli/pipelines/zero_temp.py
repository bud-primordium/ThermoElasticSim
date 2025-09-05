"""零温弹性常数场景流水线

从 YAML 配置解析材料/势、尺寸与形变点，调用库侧基准工作流
（`thermoelasticsim.elastic.benchmark`）以得到与 benchmark 一致的
图/CSV/JSON 产物。
"""

from __future__ import annotations

from ...elastic.benchmark import run_size_sweep, run_zero_temp_benchmark
from .common import get_material_by_spec, make_potential


def run_zero_temp_pipeline(cfg, outdir: str, material_symbol: str, potential_kind: str):
    """运行零温弹性常数计算。

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

    Returns
    -------
    list[dict]
        若为单尺寸，返回包含一个元素的列表；若为多尺寸扫描，返回每个尺寸的结果列表。
    """
    # 材料/势
    material_cfg = cfg.get("material", None)
    if isinstance(material_cfg, dict):
        mat = get_material_by_spec(
            material_cfg.get("symbol", material_symbol), material_cfg.get("structure")
        )
    else:
        mat = get_material_by_spec(material_symbol, cfg.get("material.structure", None))
    pot = make_potential(potential_kind)

    # 尺寸与形变点
    sizes = cfg.get("elastic.sizes", None)
    precision = bool(cfg.get("precision", False))

    uniaxial_strains = cfg.get("elastic.uniaxial_strains", None)
    shear_strains = cfg.get("elastic.shear_strains", None)
    if uniaxial_strains is None:
        amp = cfg.get("elastic.strain_amplitude", None)
        npts = cfg.get("elastic.num_points", None)
        include0 = bool(cfg.get("elastic.include_zero", True))
        if amp is not None and npts is not None:
            import numpy as np

            xs = np.linspace(-float(amp), float(amp), int(npts)).tolist()
            if not include0:
                xs = [x for x in xs if abs(x) > 1e-15]
            uniaxial_strains = xs
    if shear_strains is None:
        amp = cfg.get("elastic.shear_amplitude", None)
        npts = cfg.get("elastic.shear_num_points", None)
        include0 = bool(cfg.get("elastic.shear_include_zero", False))
        if amp is not None and npts is not None:
            import numpy as np

            xs = np.linspace(-float(amp), float(amp), int(npts)).tolist()
            if not include0:
                xs = [x for x in xs if abs(x) > 1e-15]
            shear_strains = xs

    optimizer_params = cfg.get("elastic.optimizer.params", None) or cfg.get(
        "elastic.optimizer", None
    )
    optimizer_type = cfg.get("elastic.optimizer.type", None)

    # 执行
    if sizes is None:
        sc = tuple(cfg.get("supercell", [3, 3, 3]))
        res = run_zero_temp_benchmark(
            material_params=mat,
            potential=pot,
            supercell_size=sc,
            output_dir=outdir,
            save_json=True,
            precision=precision,
            optimizer_type=optimizer_type,
            uniaxial_strains=uniaxial_strains,
            shear_strains=shear_strains,
            optimizer_params=optimizer_params,
        )
        return [res]
    else:
        results = run_size_sweep(
            sizes=[tuple(x) for x in sizes],
            output_root=outdir,
            material_params=mat,
            potential_factory=lambda: make_potential(potential_kind),
            precision=precision,
        )
        return results
