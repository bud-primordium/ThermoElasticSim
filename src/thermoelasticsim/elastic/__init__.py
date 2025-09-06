"""弹性常数计算模块"""

from .benchmark import (
    BenchmarkConfig,
    calculate_c11_c12_robust,
    calculate_c11_c12_traditional,
    calculate_c44_lammps_shear,
    run_size_sweep,
    run_zero_temp_benchmark,
)
from .deformation import Deformer
from .deformation_method import (
    ElasticConstantsSolver,
    ElasticConstantsWorkflow,
    FiniteTempElasticityWorkflow,
    ShearDeformationMethod,
    StructureRelaxer,
    ZeroTempDeformationCalculator,
)
from .materials import (
    ALUMINUM_FCC,
    CARBON_DIAMOND,
    COPPER_FCC,
    GOLD_FCC,
    MaterialParameters,
    compare_elastic_constants,
    get_all_materials,
    get_material_by_symbol,
)
from .mechanics import StrainCalculator, StressCalculator
from .wave import (
    DynamicsConfig,
    ElasticWaveAnalyzer,
    WaveExcitation,
    plot_polar_plane,
    plot_velocity_surface_3d,
    simulate_plane_wave_mvp,
)

__all__ = [
    # 形变与力学
    "Deformer",
    "StrainCalculator",
    "StressCalculator",
    # 工作流程
    "ElasticConstantsWorkflow",
    "ElasticConstantsSolver",
    "StructureRelaxer",
    "ZeroTempDeformationCalculator",
    "FiniteTempElasticityWorkflow",
    "ShearDeformationMethod",
    # 材料参数
    "MaterialParameters",
    "ALUMINUM_FCC",
    "COPPER_FCC",
    "GOLD_FCC",
    "CARBON_DIAMOND",
    "get_material_by_symbol",
    "get_all_materials",
    "compare_elastic_constants",
    # 基准工作流
    "BenchmarkConfig",
    "calculate_c11_c12_traditional",
    "calculate_c11_c12_robust",
    "calculate_c44_lammps_shear",
    "run_zero_temp_benchmark",
    "run_size_sweep",
    # 弹性波
    "ElasticWaveAnalyzer",
    "WaveExcitation",
    "DynamicsConfig",
    "simulate_plane_wave_mvp",
    "plot_polar_plane",
    "plot_velocity_surface_3d",
]
