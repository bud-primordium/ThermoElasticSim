#!/usr/bin/env python3
"""
集成轨迹记录的弹性常数计算示例

展示如何在v7现有计算流程中集成轨迹记录功能。
这个模块提供了修改后的计算方法，可以：
- 记录完整的形变过程
- 追踪优化收敛轨迹  
- 生成可用于可视化的H5轨迹文件

Author: Gilbert Young
Created: 2025-08-15
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..elastic.elastic_visualizer import ElasticVisualizer
from ..elastic.trajectory_recorder import (
    ElasticTrajectoryRecorder,
)

logger = logging.getLogger(__name__)


def calculate_c44_with_trajectory_recording(
    supercell_size: tuple[int, int, int],
    strain_magnitude: float,
    potential,  # Potential对象
    relaxer,    # StructureRelaxer对象
    output_dir: str,
    record_optimization: bool = True
) -> dict[str, Any]:
    """
    带轨迹记录的C44计算
    
    这是v7 calculate_c44_lammps_method的增强版本，
    集成了完整的轨迹记录功能。
    
    Parameters
    ----------
    supercell_size : Tuple[int, int, int]
        系统尺寸
    strain_magnitude : float
        应变幅度
    potential : Potential
        势能函数
    relaxer : StructureRelaxer
        结构弛豫器
    output_dir : str
        输出目录
    record_optimization : bool
        是否记录优化过程详细轨迹
        
    Returns
    -------
    Dict[str, Any]
        计算结果，包含轨迹文件路径
    """
    from ...elastic.mechanics import StressCalculator
    from ...utils.utils import EV_TO_GPA

    logger.info(f"开始C44计算（带轨迹记录）: {supercell_size}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建轨迹记录器
    trajectory_recorder = ElasticTrajectoryRecorder(
        output_path / "c44_trajectory.h5",
        elastic_type="C44",
        calculation_method="lammps_shear",
        supercell_size=supercell_size
    )

    # 创建系统
    cell = create_aluminum_fcc(supercell_size)

    # 应变点设置（与v7相同）
    strain_points_c44 = np.array([
        -0.004, -0.003, -0.002, -0.0015, -0.001, -0.0005, 0.0,
        0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004
    ])

    # 初始化轨迹记录器
    trajectory_recorder.initialize(
        cell=cell,
        potential=potential,
        strain_points=strain_points_c44.tolist(),
        additional_metadata={
            'strain_magnitude': strain_magnitude,
            'relaxer_type': relaxer.optimizer_type,
            'relaxer_params': relaxer.optimizer_params
        }
    )

    try:
        # 基态弛豫（记录整个过程）
        logger.info("开始基态弛豫")
        base_cell = cell.copy()

        # 记录初始状态
        trajectory_recorder.record_deformation_step(
            base_cell, 0.0, 'initial_state',
            energy=potential.calculate_energy(base_cell)
        )

        # 基态弛豫（如果需要记录优化过程）
        if record_optimization:
            relaxer_with_recording = OptimizationRecorder(relaxer, trajectory_recorder)
            relaxer_with_recording.uniform_lattice_relax(base_cell, potential)
        else:
            relaxer.uniform_lattice_relax(base_cell, potential)

        # 记录基态
        base_stress_components = StressCalculator().get_all_stress_components(base_cell, potential)
        base_stress = base_stress_components["total"] * EV_TO_GPA
        base_energy = potential.calculate_energy(base_cell)

        trajectory_recorder.record_deformation_step(
            base_cell, 0.0, 'base_state',
            stress_tensor=base_stress,
            energy=base_energy,
            converged=True
        )

        # 剪切形变计算
        directions = [4, 5, 6]  # yz, xz, xy
        direction_names = ["yz(C44)", "xz(C55)", "xy(C66)"]
        stress_indices = [(1, 2), (0, 2), (0, 1)]

        detailed_results = []

        for direction, name, (i, j) in zip(directions, direction_names, stress_indices, strict=False):
            logger.info(f"计算{name}方向")
            trajectory_recorder.set_current_strain(0.0)  # 重置应变跟踪

            strains = []
            stresses = []
            converged_states = []

            for strain in strain_points_c44:
                logger.debug(f"处理应变点: {strain:.4f}")
                trajectory_recorder.set_current_strain(strain)

                if strain == 0.0:
                    # 基态点（已记录）
                    stress_value = base_stress[i, j]
                    converged = True
                    energy = base_energy
                else:
                    # 形变点
                    deformed_cell = apply_lammps_box_shear(base_cell, direction, strain)

                    # 记录形变前状态
                    trajectory_recorder.record_deformation_step(
                        deformed_cell, strain, 'before_relax',
                        energy=potential.calculate_energy(deformed_cell),
                        converged=False
                    )

                    # 内部弛豫
                    energy_before = potential.calculate_energy(deformed_cell)

                    if record_optimization:
                        # 记录详细优化过程
                        relaxer_with_recording.set_current_strain(strain)
                        converged = relaxer_with_recording.internal_relax(deformed_cell, potential)
                    else:
                        converged = relaxer.internal_relax(deformed_cell, potential)

                    # 计算最终状态
                    energy_after = potential.calculate_energy(deformed_cell)
                    stress_components = StressCalculator().get_all_stress_components(deformed_cell, potential)
                    total_stress = stress_components["total"] * EV_TO_GPA
                    stress_value = total_stress[i, j]

                    # 记录形变后状态
                    trajectory_recorder.record_deformation_step(
                        deformed_cell, strain, 'after_relax',
                        stress_tensor=total_stress,
                        energy=energy_after,
                        converged=converged,
                        additional_data={
                            'energy_change': energy_after - energy_before,
                            'direction': name,
                            'stress_component': f'sigma_{i+1}{j+1}'
                        }
                    )

                strains.append(strain)
                stresses.append(stress_value)
                converged_states.append(converged)

                logger.debug(f"  应变={strain:+.4f}: 应力={stress_value:.4f} GPa, 收敛={converged}")

            # 计算弹性常数
            converged_strains = np.array([s for s, c in zip(strains, converged_states, strict=False) if c])
            converged_stresses = np.array([st for st, c in zip(stresses, converged_states, strict=False) if c])

            if len(converged_strains) >= 2:
                coeffs = np.polyfit(converged_strains, converged_stresses, 1)
                elastic_constant = coeffs[0]
            else:
                elastic_constant = 0.0

            detailed_results.append({
                "direction": name,
                "strains": strains,
                "stresses": stresses,
                "converged_states": converged_states,
                "elastic_constant": elastic_constant,
                "converged_count": sum(converged_states),
                "total_count": len(converged_states)
            })

            logger.info(f"{name}: {elastic_constant:.1f} GPa")

        # 完成轨迹记录
        trajectory_file = trajectory_recorder.finalize()

        # 计算最终结果
        elastic_constants = [result["elastic_constant"] for result in detailed_results]
        valid_constants = [c for c in elastic_constants if c > 0]

        if valid_constants:
            C44_cubic = np.mean(valid_constants)
            std_deviation = np.std(valid_constants) if len(valid_constants) > 1 else 0.0
        else:
            C44_cubic = 0.0
            std_deviation = 0.0

        # 生成可视化
        logger.info("生成可视化")
        visualizer = ElasticVisualizer(str(output_path))

        # 从轨迹中创建CSV数据进行可视化
        csv_data = _trajectory_to_csv_data(trajectory_file, detailed_results)
        csv_file = output_path / "elastic_data.csv"
        csv_data.to_csv(csv_file, index=False)

        # 生成可视化
        visualizer.load_csv_data(str(csv_file))
        dashboard_file = visualizer.generate_dashboard(str(output_path / "dashboard.html"))

        results = {
            "atoms": base_cell.num_atoms,
            "C44": C44_cubic,
            "elastic_constants": elastic_constants,
            "std_dev": std_deviation,
            "error_percent": (C44_cubic - 33) / 33 * 100 if C44_cubic > 0 else float("inf"),
            "detailed_results": detailed_results,
            "success": C44_cubic > 0,
            "trajectory_file": trajectory_file,
            "dashboard_file": dashboard_file,
            "csv_file": str(csv_file)
        }

        logger.info(f"C44计算完成: {C44_cubic:.1f} GPa, 轨迹文件: {trajectory_file}")
        return results

    except Exception as e:
        # 确保轨迹文件被正确关闭
        try:
            trajectory_recorder.finalize()
        except:
            pass
        logger.error(f"C44计算失败: {e}")
        raise


class OptimizationRecorder:
    """
    优化过程记录器
    
    包装现有的StructureRelaxer，添加轨迹记录功能
    """

    def __init__(self, relaxer, trajectory_recorder: ElasticTrajectoryRecorder):
        """
        初始化
        
        Parameters
        ----------
        relaxer : StructureRelaxer
            原始弛豫器
        trajectory_recorder : ElasticTrajectoryRecorder
            轨迹记录器
        """
        self.relaxer = relaxer
        self.trajectory_recorder = trajectory_recorder
        self.current_strain = 0.0

    def set_current_strain(self, strain: float):
        """设置当前应变"""
        self.current_strain = strain

    def internal_relax(self, cell, potential) -> bool:
        """带记录的内部弛豫"""
        # 创建记录回调
        def optimization_callback(iteration: int, cell_state, energy: float, forces: np.ndarray, gradient_norm: float):
            """优化过程回调函数"""
            converged = gradient_norm < self.relaxer.optimizer_params.get('gtol', 1e-7)
            self.trajectory_recorder.record_optimization_step(
                cell_state, iteration, energy, forces, gradient_norm, converged
            )

        # 执行优化（需要修改relaxer支持回调）
        # 这里简化实现，实际需要修改StructureRelaxer
        return self.relaxer.internal_relax(cell, potential)

    def uniform_lattice_relax(self, cell, potential) -> bool:
        """带记录的等比例晶格弛豫"""
        # 实际实现时需要修改StructureRelaxer支持回调
        return self.relaxer.uniform_lattice_relax(cell, potential)

    def __getattr__(self, name):
        """代理其他方法到原始relaxer"""
        return getattr(self.relaxer, name)


def _trajectory_to_csv_data(trajectory_file: str, detailed_results: list[dict]) -> 'pd.DataFrame':
    """
    将轨迹数据转换为CSV格式，用于可视化
    
    Parameters
    ----------
    trajectory_file : str
        轨迹文件路径
    detailed_results : List[Dict]
        详细计算结果
        
    Returns
    -------
    pd.DataFrame
        CSV数据
    """
    import pandas as pd

    csv_rows = []

    # 从详细结果提取数据
    for result in detailed_results:
        direction = result["direction"]
        strains = result["strains"]
        stresses = result["stresses"]
        converged_states = result["converged_states"]

        # 确定计算方法和方向
        if "C44" in direction:
            method = "C44_shear"
            strain_dir = "yz"
            stress_dir = "yz"
        elif "C55" in direction:
            method = "C55_shear"
            strain_dir = "xz"
            stress_dir = "xz"
        elif "C66" in direction:
            method = "C66_shear"
            strain_dir = "xy"
            stress_dir = "xy"
        else:
            method = "shear"
            strain_dir = "unknown"
            stress_dir = "unknown"

        for strain, stress, converged in zip(strains, stresses, converged_states, strict=False):
            csv_row = {
                "calculation_method": method,
                "applied_strain_direction": strain_dir,
                "measured_stress_direction": stress_dir,
                "applied_strain": strain,
                "measured_stress_GPa": stress,
                "optimization_converged": converged,
                "is_reference_state": (strain == 0.0),
                "optimization_status": "SUCCESS" if converged else "FAILED"
            }
            csv_rows.append(csv_row)

    return pd.DataFrame(csv_rows)


def create_aluminum_fcc(supercell_size=(3, 3, 3)):
    """创建FCC铝系统（从v7复制）"""
    from ...core.structure import Atom, Cell

    a = 4.045  # EAM Al1文献值
    nx, ny, nz = supercell_size
    lattice = np.array(
        [[a * nx, 0, 0], [0, a * ny, 0], [0, 0, a * nz]], dtype=np.float64
    )

    atoms = []
    atom_id = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                positions = [
                    [i * a, j * a, k * a],
                    [i * a + a / 2, j * a + a / 2, k * a],
                    [i * a + a / 2, j * a, k * a + a / 2],
                    [i * a, j * a + a / 2, k * a + a / 2],
                ]
                for pos in positions:
                    atoms.append(
                        Atom(
                            id=atom_id,
                            symbol="Al",
                            mass_amu=26.9815,
                            position=np.array(pos),
                        )
                    )
                    atom_id += 1

    return Cell(lattice, atoms, pbc_enabled=True)


def apply_lammps_box_shear(cell, direction, strain_magnitude):
    """LAMMPS风格盒子剪切（从v7复制）"""
    from ...core.structure import Atom, Cell

    lattice = cell.lattice_vectors.copy()
    positions = cell.get_positions().copy()

    if direction == 4:  # yz剪切 → σ23
        lattice[2, 1] += strain_magnitude * lattice[2, 2]
        for i, pos in enumerate(positions):
            positions[i, 1] += strain_magnitude * pos[2]
    elif direction == 5:  # xz剪切 → σ13
        lattice[2, 0] += strain_magnitude * lattice[2, 2]
        for i, pos in enumerate(positions):
            positions[i, 0] += strain_magnitude * pos[2]
    elif direction == 6:  # xy剪切 → σ12
        lattice[1, 0] += strain_magnitude * lattice[1, 1]
        for i, pos in enumerate(positions):
            positions[i, 0] += strain_magnitude * pos[1]
    else:
        raise ValueError(f"不支持的剪切方向: {direction}")

    new_cell = Cell(
        lattice,
        [
            Atom(id=i, symbol="Al", mass_amu=26.9815, position=pos)
            for i, pos in enumerate(positions)
        ],
        pbc_enabled=True,
    )

    return new_cell


def demonstrate_enhanced_elastic_calculation():
    """
    演示如何使用增强的弹性常数计算
    
    这个函数展示了完整的工作流：
    1. 计算带轨迹记录
    2. 自动生成可视化
    3. 导出分析结果
    """
    from ...elastic.deformation_method.zero_temp import StructureRelaxer
    from ...potentials.eam import EAMAl1Potential

    # 初始化
    potential = EAMAl1Potential(cutoff=6.5)
    relaxer = StructureRelaxer(
        optimizer_type="L-BFGS",
        optimizer_params={
            "ftol": 1e-7,
            "gtol": 1e-6,
            "maxiter": 2000
        }
    )

    # 运行增强计算
    results = calculate_c44_with_trajectory_recording(
        supercell_size=(3, 3, 3),
        strain_magnitude=0.001,
        potential=potential,
        relaxer=relaxer,
        output_dir="enhanced_c44_calculation",
        record_optimization=True
    )

    print("增强C44计算完成!")
    print(f"C44 = {results['C44']:.1f} GPa")
    print(f"轨迹文件: {results['trajectory_file']}")
    print(f"可视化仪表板: {results['dashboard_file']}")

    return results


if __name__ == "__main__":
    # 演示用法
    demonstrate_enhanced_elastic_calculation()
