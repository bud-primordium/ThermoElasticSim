#!/usr/bin/env python3
"""
弹性常数基准测试：零温下FCC铝的系统尺寸效应研究

本脚本实现了基于分子动力学的弹性常数计算，特别针对FCC铝体系的C11、C12和C44弹性常数。
采用多种系统尺寸进行基准测试，评估尺寸效应对计算精度的影响。

主要功能：
- FCC铝系统的多尺寸弹性常数计算（C11、C12、C44）
- LAMMPS风格的盒子剪切变形方法
- 轨迹记录和可视化动画生成
- 综合质量评估和结果对比

计算方法：
- C11/C12：单轴应变法，同时测量同轴和横向应力响应
- C44：剪切应变法，使用yz、xz、xy三个独立剪切模式
- 基态优化：等比例晶格弛豫，保持FCC对称性

技术特点：
- 高精度数值优化器配置
- 完整的应力张量计算
- 多点应变测试和线性拟合
- 立方对称化处理
- 详细的收敛性分析

适用范围：
- 材料科学中的弹性性质研究
- 分子动力学方法验证
- 势函数参数校准
- 系统尺寸效应分析

创建时间：2025年8月
"""

import logging
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体避免乱码
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 添加src路径
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from thermoelasticsim.core.structure import Atom, Cell
from thermoelasticsim.elastic.deformation_method.zero_temp import (
    StructureRelaxer,
    ZeroTempDeformationCalculator,
)
from thermoelasticsim.potentials.eam import EAMAl1Potential
from thermoelasticsim.utils.modern_visualization import ModernVisualizer
from thermoelasticsim.utils.utils import EV_TO_GPA
from thermoelasticsim.visualization.elastic.response_plotter import ResponsePlotter
from thermoelasticsim.visualization.elastic.trajectory_recorder import (
    ElasticTrajectoryRecorder,
)


def setup_logging(test_name: str = "c44_final_v7"):
    """设置日志系统，创建独立运行目录"""
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
    logger.setLevel(logging.DEBUG)  # 显示详细信息

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 文件handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 控制台显示info级别
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_filepath, run_dir  # 返回运行目录用于保存其他文件


def create_aluminum_fcc(supercell_size=(3, 3, 3)):
    """创建FCC铝系统（标准方法）"""
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
    """
    LAMMPS风格盒子剪切 - 关键突破方法

    Parameters
    ----------
    cell : Cell
        要形变的系统
    direction : int
        剪切方向：4(yz), 5(xz), 6(xy)
    strain_magnitude : float
        应变幅度

    Returns
    -------
    Cell
        形变后的系统
    """
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


def plot_stress_strain_response(
    supercell_size, detailed_results, strain_magnitude, run_dir
):
    """
    生成C44/C55/C66分别的剪切应力-应变响应关系图，每个剪切模式单独子图
    """
    fig = plt.figure(figsize=(18, 12))

    # 创建2行2列子图布局：上排3个剪切子图，下排1个汇总对比图
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3, height_ratios=[3, 2])
    ax_shear = [fig.add_subplot(gs[0, i]) for i in range(3)]  # 上排3个剪切子图
    ax_summary = fig.add_subplot(gs[1, :])  # 下排汇总图

    # 准备数据
    directions = ["yz(C44)", "xz(C55)", "xy(C66)"]
    colors = ["#2E86C1", "#E74C3C", "#58D68D"]
    markers = ["o", "s", "^"]
    literature_values = [33.0, 33.0, 33.0]  # GPa，立方对称材料C44=C55=C66

    # 为每个剪切模式单独绘制子图
    for i, result in enumerate(detailed_results):
        ax = ax_shear[i]
        direction = result["direction"]
        color = colors[i]
        marker = markers[i]
        lit_value = literature_values[i]

        # 获取多点数据
        strains = result["strains"]
        stresses = result["stresses"]
        converged_states = result["converged_states"]

        # 分别绘制收敛和不收敛的点
        converged_strains = [
            s for s, c in zip(strains, converged_states, strict=False) if c
        ]
        converged_stresses = [
            st for st, c in zip(stresses, converged_states, strict=False) if c
        ]
        failed_strains = [
            s for s, c in zip(strains, converged_states, strict=False) if not c
        ]
        failed_stresses = [
            st for st, c in zip(stresses, converged_states, strict=False) if not c
        ]

        # 收敛点：实心符号
        if converged_strains:
            ax.scatter(
                converged_strains,
                converged_stresses,
                marker=marker,
                color=color,
                s=100,
                label="收敛点",
                alpha=0.8,
                edgecolors="black",
                linewidth=1,
            )

        # 不收敛点：空心符号
        if failed_strains:
            ax.scatter(
                failed_strains,
                failed_stresses,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                s=100,
                label="未收敛",
                alpha=0.8,
                linewidth=2,
            )

        # 添加文献值理论斜率参考线
        if strains:
            strain_range = np.linspace(min(strains) * 1.2, max(strains) * 1.2, 100)
            theory_stress = lit_value * strain_range
            ax.plot(
                strain_range,
                theory_stress,
                "k:",
                linewidth=2,
                alpha=0.7,
                label=f"理论斜率 ({lit_value} GPa)",
            )

        # 只对收敛点进行线性拟合
        if len(converged_strains) >= 2:
            # 线性拟合
            coeffs = np.polyfit(converged_strains, converged_stresses, 1)
            fit_strains = np.linspace(
                min(converged_strains), max(converged_strains), 100
            )
            fit_stresses = np.polyval(coeffs, fit_strains)
            ax.plot(
                fit_strains,
                fit_stresses,
                "--",
                color=color,
                alpha=0.9,
                linewidth=3,
                label=f"拟合 ({coeffs[0]:.1f} GPa)",
            )

            # 计算R²
            y_pred = np.polyval(coeffs, converged_strains)
            ss_res = np.sum((converged_stresses - y_pred) ** 2)
            ss_tot = np.sum((converged_stresses - np.mean(converged_stresses)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            # 在图上显示拟合质量
            ax.text(
                0.05,
                0.95,
                f"R² = {r_squared:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # 设置单个子图属性
        shear_component = direction.split("(")[0]  # yz, xz, xy
        ax.set_xlabel(f"{shear_component}剪切应变", fontsize=12)
        ax.set_ylabel(f"{shear_component}剪切应力 (GPa)", fontsize=12)
        ax.set_title(f"{direction}", fontsize=14, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")

    # 下方汇总图：弹性常数对比
    elastic_constants = []
    convergence_quality = []
    for result in detailed_results:
        elastic_constants.append(result["elastic_constant"])
        # 计算收敛质量（收敛点比例）
        convergence_quality.append(
            sum(result["converged_states"]) / len(result["converged_states"])
        )

    literature_value = 33.0  # GPa
    x_pos = np.arange(len(directions))
    bars = ax_summary.bar(
        x_pos,
        elastic_constants,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # 根据收敛质量调整透明度
    for bar, quality in zip(bars, convergence_quality, strict=False):
        bar.set_alpha(0.3 + 0.7 * quality)  # 收敛质量高的更不透明

    # 文献值参考线
    ax_summary.axhline(
        y=literature_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"文献值 ({literature_value} GPa)",
    )

    # 添加数值标签和收敛质量
    for i, (bar, value, quality) in enumerate(
        zip(bars, elastic_constants, convergence_quality, strict=False)
    ):
        height = bar.get_height()
        ax_summary.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(height * 0.02, 2),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

        # 计算误差
        error = (value - literature_value) / literature_value * 100
        ax_summary.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(height * 0.08, 8),
            f"({error:+.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="gray",
        )

        # 显示收敛率
        ax_summary.text(
            bar.get_x() + bar.get_width() / 2.0,
            height / 2,
            f"{quality:.0%}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

    ax_summary.set_xlabel("剪切模式", fontsize=12)
    ax_summary.set_ylabel("弹性常数 (GPa)", fontsize=12)
    ax_summary.set_title(
        f"弹性常数汇总 - {supercell_size[0]}×{supercell_size[1]}×{supercell_size[2]}系统\n平均值: {np.mean(elastic_constants):.1f} GPa",
        fontsize=14,
    )
    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(directions)
    ax_summary.grid(True, alpha=0.3, axis="y")
    ax_summary.legend(fontsize=10)

    # 设置y轴范围确保能看到所有数据
    max_val = max(max(elastic_constants), literature_value)
    ax_summary.set_ylim(0, max_val * 1.3)

    plt.tight_layout()

    # 🔧 修复文件命名：明确标识为剪切计算
    filename = f"c44_c55_c66_shear_response_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.png"
    filepath = os.path.join(run_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()  # 释放内存

    print(f"  📊 C44剪切响应图已保存: {filename}")
    return filename


def calculate_c44_lammps_method(
    supercell_size, strain_magnitude, potential, relaxer, run_dir
):
    """
    使用LAMMPS方法计算C44，多点应变测试
    集成轨迹记录和动画生成功能

    Returns
    -------
    dict : 包含C44计算结果和诊断信息
    """
    logger = logging.getLogger(__name__)

    # 创建系统
    cell = create_aluminum_fcc(supercell_size)

    # 初始化轨迹记录器
    trajectory_file = os.path.join(
        run_dir,
        f"c44_trajectory_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.h5",
    )
    recorder = ElasticTrajectoryRecorder(
        trajectory_file, "C44", "shear_lammps", supercell_size
    )

    # 应变点定义
    strain_points_c44 = np.array(
        [
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
        ]
    )

    # 初始化轨迹记录
    recorder.initialize(cell, potential, strain_points_c44.tolist())

    # 基态弛豫 - 使用等比例晶格弛豫（更快且保持对称性）
    base_cell = cell.copy()

    # 创建支持轨迹记录的relaxer
    enhanced_relaxer = StructureRelaxer(
        optimizer_type=relaxer.optimizer_type,
        optimizer_params=relaxer.optimizer_params,
        supercell_dims=relaxer.supercell_dims,
        trajectory_recorder=recorder,
    )

    enhanced_relaxer.uniform_lattice_relax(base_cell, potential)
    base_stress = base_cell.calculate_stress_tensor(potential)

    # 记录基态
    recorder.record_deformation_step(
        base_cell,
        0.0,
        "base_state",
        stress_tensor=base_stress,
        energy=potential.calculate_energy(base_cell),
        converged=True,
    )

    # 基态诊断
    base_stress_magnitude = np.linalg.norm(base_stress * EV_TO_GPA)
    lattice = base_cell.lattice_vectors
    off_diagonal = np.array([lattice[0, 1], lattice[0, 2], lattice[1, 2]])
    asymmetry = np.max(np.abs(off_diagonal))

    logger.info(
        f"基态诊断: 应力={base_stress_magnitude:.4f} GPa, 非对称性={asymmetry:.2e} Å"
    )

    # 三个剪切方向
    directions = [4, 5, 6]
    direction_names = ["yz(C44)", "xz(C55)", "xy(C66)"]
    stress_indices = [(1, 2), (0, 2), (0, 1)]

    elastic_constants = []
    detailed_results = []
    csv_data_all = []  # 存储所有CSV数据

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 暂未使用

    for direction, name, (i, j) in zip(
        directions, direction_names, stress_indices, strict=False
    ):
        logger.debug(f"开始计算 {name} 方向，共{len(strain_points_c44)}个应变点")

        strains = []
        stresses = []
        converged_states = []
        csv_data = []  # 当前方向的CSV数据

        for strain in strain_points_c44:
            # 设置轨迹记录器当前应变
            recorder.set_current_strain(strain)

            if strain == 0.0:
                # 基态已经记录过了，跳过
                stress_value = base_stress[i, j] * EV_TO_GPA
                converged = True
                energy = potential.calculate_energy(base_cell)

                csv_row = {
                    "method": "C44_shear",
                    "direction": name.split("(")[0],
                    "strain": strain,
                    "stress_GPa": stress_value,
                    "energy_eV": energy,
                    "converged": converged,
                    "is_base": True,
                }
            else:
                # 施加形变
                deformed_cell = apply_lammps_box_shear(base_cell, direction, strain)

                # 内部弛豫 - enhanced_relaxer会自动记录轨迹
                converged = enhanced_relaxer.internal_relax(deformed_cell, potential)

                # 获取最终状态
                energy_after = potential.calculate_energy(deformed_cell)
                stress_after = deformed_cell.calculate_stress_tensor(potential)
                stress_value = stress_after[i, j] * EV_TO_GPA

                csv_row = {
                    "method": "C44_shear",
                    "direction": name.split("(")[0],
                    "strain": strain,
                    "stress_GPa": stress_value,
                    "energy_eV": energy_after,
                    "converged": converged,
                    "is_base": False,
                }

            strains.append(strain)
            stresses.append(stress_value)
            converged_states.append(converged)
            csv_data.append(csv_row)
            csv_data_all.append(csv_row)

            logger.debug(
                f"  应变={strain:+.4f}: 应力={stress_value:.4f} GPa, 收敛={converged}, 能量={csv_row.get('energy_eV', 'N/A'):.6f} eV"
            )

        # 只记录到汇总数据，不单独保存

        # 只用收敛点计算弹性常数
        converged_strains = np.array(
            [s for s, c in zip(strains, converged_states, strict=False) if c]
        )
        converged_stresses = np.array(
            [st for st, c in zip(stresses, converged_states, strict=False) if c]
        )

        if len(converged_strains) >= 2:
            # 线性拟合：stress = base_stress + elastic_constant * strain
            coeffs = np.polyfit(converged_strains, converged_stresses, 1)
            elastic_constant = coeffs[0]  # 斜率就是弹性常数
        else:
            logger.warning(
                f"{name} 方向收敛点不足({len(converged_strains)}个)，使用fallback计算"
            )
            elastic_constant = 0.0  # 失败情况

        elastic_constants.append(elastic_constant)

        # 详细记录（包含多点数据）
        detailed_results.append(
            {
                "direction": name,
                "base_stress": base_stress[i, j] * EV_TO_GPA,
                "strains": strains,
                "stresses": stresses,
                "converged_states": converged_states,
                "elastic_constant": elastic_constant,
                "converged_count": sum(converged_states),
                "total_count": len(converged_states),
                "csv_file": "included_in_summary",
            }
        )

        convergence_rate = sum(converged_states) / len(converged_states)
        logger.info(
            f"{name}: {elastic_constant:.1f} GPa (收敛率: {convergence_rate:.1%}, {sum(converged_states)}/{len(converged_states)}点)"
        )

    # 保存汇总CSV数据到运行目录（只要一个合并的输出）
    csv_summary_filename = f"c44_shear_analysis_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.csv"
    csv_summary_filepath = os.path.join(run_dir, csv_summary_filename)

    # 更新键名以符合要求
    df_summary = pd.DataFrame(csv_data_all)
    # 重命名列以更清晰的表示
    column_mapping = {
        "method": "calculation_method",
        "strain_direction": "applied_strain_direction",
        "stress_direction": "measured_stress_direction",
        "strain": "applied_strain",
        "stress_GPa": "measured_stress_GPa",
        "energy_eV": "total_energy_eV",
        "converged": "optimization_converged",
        "base_state": "is_reference_state",
        "optimization_details": "optimization_status",
    }
    df_summary = df_summary.rename(columns=column_mapping)
    df_summary.to_csv(csv_summary_filepath, index=False)
    logger.info(f"💾 C44剪切分析数据已保存: {csv_summary_filename}")
    print(f"  📋 C44剪切分析CSV: {csv_summary_filename}")

    # 立方对称化（只使用有效的弹性常数）
    valid_constants = [c for c in elastic_constants if c > 0]
    if valid_constants:
        C44_cubic = np.mean(valid_constants)
        std_deviation = np.std(valid_constants) if len(valid_constants) > 1 else 0.0
    else:
        C44_cubic = 0.0
        std_deviation = 0.0

    # 质量评估
    strain_ok = strain_magnitude < 0.005
    size_ok = base_cell.num_atoms >= 100
    stress_ok = base_stress_magnitude < 0.1
    consistency_ok = std_deviation < 5.0
    accuracy_ok = abs((C44_cubic - 33) / 33 * 100) < 50 if C44_cubic > 0 else False

    success_score = (
        sum([strain_ok, size_ok, stress_ok, consistency_ok, accuracy_ok]) / 5
    )

    # 生成应力-应变响应关系图
    plot_filename = plot_stress_strain_response(
        supercell_size, detailed_results, strain_magnitude, run_dir
    )

    # 完成轨迹记录
    trajectory_path = recorder.finalize()

    # 生成轨迹动画
    try:
        visualizer = ModernVisualizer()

        # 生成交互式HTML动画
        animation_html = os.path.join(
            run_dir,
            f"c44_trajectory_animation_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.html",
        )
        visualizer.create_trajectory_animation_plotly(
            trajectory_path, animation_html, skip=2, duration=500
        )

        # 生成GIF动画（较快）
        animation_gif = os.path.join(
            run_dir,
            f"c44_trajectory_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.gif",
        )
        visualizer.create_trajectory_video(
            trajectory_path, animation_gif, fps=5, skip=3, dpi=80, figsize=(8, 6)
        )

        logger.info(
            f"轨迹动画生成完成: {os.path.basename(animation_html)}, {os.path.basename(animation_gif)}"
        )
        print(f"  🎬 轨迹动画: {os.path.basename(animation_html)}")
        print(f"  📱 轨迹GIF: {os.path.basename(animation_gif)}")

        animation_files = {
            "html": os.path.basename(animation_html),
            "gif": os.path.basename(animation_gif),
        }

    except Exception as e:
        logger.warning(f"动画生成失败: {e}")
        print(f"  ⚠️ 动画生成失败: {e}")
        animation_files = {}

    # 使用新的ResponsePlotter生成增强的拟合图（替换原有图）
    try:
        plotter = ResponsePlotter()
        enhanced_plot = os.path.join(
            run_dir,
            f"c44_enhanced_response_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.png",
        )
        enhanced_filename = plotter.plot_shear_response(
            detailed_results, supercell_size, enhanced_plot
        )
        logger.info(f"增强拟合图生成完成: {enhanced_filename}")
        print(f"  📈 增强拟合图: {enhanced_filename}")

        # 确保删除原有的重复图（使用原始文件名）
        original_plot_path = os.path.join(run_dir, plot_filename)
        if os.path.exists(original_plot_path) and original_plot_path != enhanced_plot:
            try:
                os.remove(original_plot_path)
                logger.info(f"删除重复图: {original_plot_path}")
                print(f"  🗑️ 删除重复图: {os.path.basename(original_plot_path)}")
            except Exception as e:
                logger.warning(f"删除重复图失败: {e}")

        plot_filename = os.path.basename(enhanced_filename)  # 统一使用增强图

    except Exception as e:
        logger.warning(f"增强拟合图生成失败: {e}")
        enhanced_filename = plot_filename

    return {
        "atoms": base_cell.num_atoms,
        "C44": C44_cubic,
        "elastic_constants": elastic_constants,
        "std_dev": std_deviation,
        "error_percent": (C44_cubic - 33) / 33 * 100 if C44_cubic > 0 else float("inf"),
        "base_stress_magnitude": base_stress_magnitude,
        "asymmetry": asymmetry,
        "success_score": success_score,
        "quality_checks": {
            "strain_range": strain_ok,
            "system_size": size_ok,
            "stress_convergence": stress_ok,
            "consistency": consistency_ok,
            "accuracy": accuracy_ok,
        },
        "detailed_results": detailed_results,
        "success": C44_cubic > 0,
        "plot_file": plot_filename,  # 统一的拟合图文件
        "csv_file": csv_summary_filename,
        "trajectory_file": os.path.basename(trajectory_path),
        "animation_files": animation_files,
    }


def plot_c12_stress_strain_response(supercell_size, csv_data, run_dir):
    """
    生成C12交叉应力-应变响应关系图 (yy应变→xx应力)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 准备数据
    strains = [row["strain"] for row in csv_data]
    stresses = [row["stress_GPa"] for row in csv_data]
    converged_states = [row["converged"] for row in csv_data]

    # 分别绘制收敛和不收敛的点
    converged_strains = [
        s for s, c in zip(strains, converged_states, strict=False) if c
    ]
    converged_stresses = [
        st for st, c in zip(stresses, converged_states, strict=False) if c
    ]
    failed_strains = [
        s for s, c in zip(strains, converged_states, strict=False) if not c
    ]
    failed_stresses = [
        st for st, c in zip(stresses, converged_states, strict=False) if not c
    ]

    # 左图：应力-应变关系
    if converged_strains:
        ax1.scatter(
            converged_strains,
            converged_stresses,
            marker="s",
            color="#E74C3C",
            s=80,
            label="C12交叉 (收敛)",
            alpha=0.8,
            edgecolors="black",
        )

    if failed_strains:
        ax1.scatter(
            failed_strains,
            failed_stresses,
            marker="s",
            facecolors="none",
            edgecolors="#E74C3C",
            s=80,
            label="C12交叉 (未收敛)",
            alpha=0.8,
            linewidth=2,
        )

    # 添加文献值理论斜率参考线
    literature_C12 = 61.0  # GPa
    strain_range = np.linspace(-0.003, 0.003, 100)
    theory_stress = literature_C12 * strain_range
    ax1.plot(
        strain_range,
        theory_stress,
        "k:",
        linewidth=2,
        alpha=0.7,
        label=f"理论斜率 (C12={literature_C12} GPa)",
    )

    # 线性拟合（只用收敛点）
    if len(converged_strains) >= 2:
        coeffs = np.polyfit(converged_strains, converged_stresses, 1)
        fit_strains = np.linspace(min(converged_strains), max(converged_strains), 100)
        fit_stresses = np.polyval(coeffs, fit_strains)
        ax1.plot(
            fit_strains, fit_stresses, "--", color="#E74C3C", alpha=0.7, linewidth=2
        )
        C12_fitted = coeffs[0]
    else:
        C12_fitted = 0.0

    ax1.set_xlabel("单轴应变 εyy", fontsize=12)
    ax1.set_ylabel("交叉应力 σxx (GPa)", fontsize=12)
    ax1.set_title(
        f"{supercell_size[0]}×{supercell_size[1]}×{supercell_size[2]} 系统\nC12交叉应力-应变响应",
        fontsize=13,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="best")

    # 右图：弹性常数对比
    literature_C12 = 61.0  # GPa
    convergence_rate = sum(converged_states) / len(converged_states)

    bar = ax2.bar(
        ["C12"],
        [C12_fitted],
        color="#E74C3C",
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # 根据收敛质量调整透明度
    bar[0].set_alpha(0.3 + 0.7 * convergence_rate)

    # 文献值参考线
    ax2.axhline(
        y=literature_C12,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"文献值 ({literature_C12} GPa)",
    )

    # 添加数值标签
    if C12_fitted != 0.0:
        height = bar[0].get_height()
        ax2.text(
            0,
            height + max(height * 0.02, 2),
            f"{C12_fitted:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

        # 计算误差
        error = (C12_fitted - literature_C12) / literature_C12 * 100
        ax2.text(
            0,
            height + max(height * 0.08, 8),
            f"({error:+.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="gray",
        )

        # 显示收敛率
        ax2.text(
            0,
            height / 2,
            f"{convergence_rate:.0%}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

    ax2.set_ylabel("弹性常数 (GPa)", fontsize=12)
    ax2.set_title(f"C12计算结果\n{C12_fitted:.1f} GPa", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=10)

    # 设置y轴范围
    max_val = max(C12_fitted if C12_fitted != 0.0 else 0, literature_C12)
    ax2.set_ylim(0, max_val * 1.3)

    plt.tight_layout()

    # 保存图片到运行目录
    filename = f"c12_stress_strain_response_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.png"
    filepath = os.path.join(run_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    # plt.show()  # 注释掉避免弹出

    print(f"  📊 C12应力-应变图已保存: {filename}")
    return filename


def plot_c11_stress_strain_response(supercell_size, csv_data, run_dir):
    """
    生成C11单轴应力-应变响应关系图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 准备数据
    strains = [row["strain"] for row in csv_data]
    stresses = [row["stress_GPa"] for row in csv_data]
    converged_states = [row["converged"] for row in csv_data]

    # 分别绘制收敛和不收敛的点
    converged_strains = [
        s for s, c in zip(strains, converged_states, strict=False) if c
    ]
    converged_stresses = [
        st for st, c in zip(stresses, converged_states, strict=False) if c
    ]
    failed_strains = [
        s for s, c in zip(strains, converged_states, strict=False) if not c
    ]
    failed_stresses = [
        st for st, c in zip(stresses, converged_states, strict=False) if not c
    ]

    # 左图：应力-应变关系
    if converged_strains:
        ax1.scatter(
            converged_strains,
            converged_stresses,
            marker="o",
            color="#2E86C1",
            s=80,
            label="C11单轴 (收敛)",
            alpha=0.8,
            edgecolors="black",
        )

    if failed_strains:
        ax1.scatter(
            failed_strains,
            failed_stresses,
            marker="o",
            facecolors="none",
            edgecolors="#2E86C1",
            s=80,
            label="C11单轴 (未收敛)",
            alpha=0.8,
            linewidth=2,
        )

    # 添加文献值理论斜率参考线
    literature_C11 = 110.0  # GPa
    strain_range = np.linspace(-0.003, 0.003, 100)
    theory_stress = literature_C11 * strain_range
    ax1.plot(
        strain_range,
        theory_stress,
        "k:",
        linewidth=2,
        alpha=0.7,
        label=f"理论斜率 (C11={literature_C11} GPa)",
    )

    # 线性拟合（只用收敛点）
    if len(converged_strains) >= 2:
        coeffs = np.polyfit(converged_strains, converged_stresses, 1)
        fit_strains = np.linspace(min(converged_strains), max(converged_strains), 100)
        fit_stresses = np.polyval(coeffs, fit_strains)
        ax1.plot(
            fit_strains, fit_stresses, "--", color="#2E86C1", alpha=0.7, linewidth=2
        )
        C11_fitted = coeffs[0]
    else:
        C11_fitted = 0.0

    ax1.set_xlabel("单轴应变 εxx", fontsize=12)
    ax1.set_ylabel("单轴应力 σxx (GPa)", fontsize=12)
    ax1.set_title(
        f"{supercell_size[0]}×{supercell_size[1]}×{supercell_size[2]} 系统\nC11单轴应力-应变响应",
        fontsize=13,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="best")

    # 右图：弹性常数对比
    literature_C11 = 110.0  # GPa
    convergence_rate = sum(converged_states) / len(converged_states)

    bar = ax2.bar(
        ["C11"],
        [C11_fitted],
        color="#2E86C1",
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
    )

    # 根据收敛质量调整透明度
    bar[0].set_alpha(0.3 + 0.7 * convergence_rate)

    # 文献值参考线
    ax2.axhline(
        y=literature_C11,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"文献值 ({literature_C11} GPa)",
    )

    # 添加数值标签
    if C11_fitted > 0:
        height = bar[0].get_height()
        ax2.text(
            0,
            height + max(height * 0.02, 2),
            f"{C11_fitted:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

        # 计算误差
        error = (C11_fitted - literature_C11) / literature_C11 * 100
        ax2.text(
            0,
            height + max(height * 0.08, 8),
            f"({error:+.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color="gray",
        )

        # 显示收敛率
        ax2.text(
            0,
            height / 2,
            f"{convergence_rate:.0%}",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

    ax2.set_ylabel("弹性常数 (GPa)", fontsize=12)
    ax2.set_title(f"C11计算结果\n{C11_fitted:.1f} GPa", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=10)

    # 设置y轴范围
    max_val = max(C11_fitted if C11_fitted > 0 else 0, literature_C11)
    ax2.set_ylim(0, max_val * 1.3)

    plt.tight_layout()

    # 保存图片到运行目录
    filename = f"c11_stress_strain_response_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.png"
    filepath = os.path.join(run_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    # plt.show()  # 注释掉避免弹出

    print(f"  📊 C11应力-应变图已保存: {filename}")
    return filename


def calculate_c11_c12_combined_method(supercell_size, potential, relaxer, run_dir):
    """
    高效联合计算C11/C12：一次单轴应变同时得到C11和C12

    原理：施加xx应变，测量：
    - σxx → C11 (同轴应力)
    - σyy, σzz → C12 (横向应力)

    这样可以大大提高数据利用效率
    """
    logger = logging.getLogger(__name__)

    # 创建系统
    cell = create_aluminum_fcc(supercell_size)

    print("\n高效联合计算C11/C12 - 一次应变双重收获")
    print(
        f"系统: {supercell_size[0]}×{supercell_size[1]}×{supercell_size[2]} ({cell.num_atoms}原子)"
    )

    # 基态弛豫 - 使用等比例晶格弛豫（更快且保持对称性）
    base_cell = cell.copy()
    relaxer.uniform_lattice_relax(base_cell, potential)
    base_energy = potential.calculate_energy(base_cell)

    logger.info("C11/C12联合基态诊断: 系统设置完成")

    # 相同的应变点（与C44一致）
    strain_points = np.array(
        [-0.003, -0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002, 0.003]
    )

    c11_data_all = []  # C11数据
    c12_data_all = []  # C12数据

    # 施加xx方向单轴应变，同时获得C11和C12
    for strain in strain_points:
        if strain == 0.0:
            # 基态点 - 完整应力分析
            from thermoelasticsim.elastic.mechanics import StressCalculator

            stress_calc = StressCalculator()
            stress_components = stress_calc.get_all_stress_components(
                base_cell, potential
            )

            total_stress = stress_components["total"] * EV_TO_GPA

            # C11: xx应变 → xx应力
            stress_xx = total_stress[0, 0]
            # C12: xx应变 → yy应力 (或zz应力，应该相同)
            stress_yy = total_stress[1, 1]
            # stress_zz = total_stress[2, 2]  # 已在记录中直接使用total_stress[2,2]

            converged = True
            energy = base_energy

            # C11数据记录
            c11_row = {
                "calculation_method": "C11_uniaxial_combined",
                "applied_strain_direction": "xx",
                "measured_stress_direction": "xx",
                "applied_strain": strain,
                "measured_stress_GPa": stress_xx,
                "stress_total_xx_GPa": total_stress[0, 0],
                "stress_total_yy_GPa": total_stress[1, 1],
                "stress_total_zz_GPa": total_stress[2, 2],
                "stress_total_xy_GPa": total_stress[0, 1],
                "stress_total_xz_GPa": total_stress[0, 2],
                "stress_total_yz_GPa": total_stress[1, 2],
                "total_energy_eV": energy,
                "optimization_converged": converged,
                "is_reference_state": True,
                "optimization_status": "Base state (uniform lattice relaxed)",
            }

            # C12数据记录 - 使用yy应力
            c12_row = {
                "calculation_method": "C12_cross_combined",
                "applied_strain_direction": "xx",
                "measured_stress_direction": "yy",
                "applied_strain": strain,
                "measured_stress_GPa": stress_yy,
                "stress_total_xx_GPa": total_stress[0, 0],
                "stress_total_yy_GPa": total_stress[1, 1],
                "stress_total_zz_GPa": total_stress[2, 2],
                "stress_total_xy_GPa": total_stress[0, 1],
                "stress_total_xz_GPa": total_stress[0, 2],
                "stress_total_yz_GPa": total_stress[1, 2],
                "total_energy_eV": energy,
                "optimization_converged": converged,
                "is_reference_state": True,
                "optimization_status": "Base state (uniform lattice relaxed)",
            }
        else:
            # 单轴应变点
            deformed_cell = apply_volume_strain(base_cell, strain)

            # 记录变形前能量
            energy_before = potential.calculate_energy(deformed_cell)

            # 尝试内部弛豫
            converged = relaxer.internal_relax(deformed_cell, potential)

            # 记录变形后状态 - 完整应力分析
            energy_after = potential.calculate_energy(deformed_cell)

            # 获取完整应力分析
            from thermoelasticsim.elastic.mechanics import StressCalculator

            stress_calc = StressCalculator()
            stress_components = stress_calc.get_all_stress_components(
                deformed_cell, potential
            )

            total_stress = stress_components["total"] * EV_TO_GPA

            # C11: xx应变 → xx应力
            stress_xx = total_stress[0, 0]
            # C12: xx应变 → yy应力
            stress_yy = total_stress[1, 1]

            # C11数据记录
            c11_row = {
                "calculation_method": "C11_uniaxial_combined",
                "applied_strain_direction": "xx",
                "measured_stress_direction": "xx",
                "applied_strain": strain,
                "measured_stress_GPa": stress_xx,
                "stress_total_xx_GPa": total_stress[0, 0],
                "stress_total_yy_GPa": total_stress[1, 1],
                "stress_total_zz_GPa": total_stress[2, 2],
                "stress_total_xy_GPa": total_stress[0, 1],
                "stress_total_xz_GPa": total_stress[0, 2],
                "stress_total_yz_GPa": total_stress[1, 2],
                "total_energy_eV": energy_after,
                "energy_before_relax_eV": energy_before,
                "energy_change_eV": energy_after - energy_before,
                "optimization_converged": converged,
                "is_reference_state": False,
                "optimization_status": f"Internal relax: {'SUCCESS' if converged else 'FAILED'}",
            }

            # C12数据记录
            c12_row = {
                "calculation_method": "C12_cross_combined",
                "applied_strain_direction": "xx",
                "measured_stress_direction": "yy",
                "applied_strain": strain,
                "measured_stress_GPa": stress_yy,
                "stress_total_xx_GPa": total_stress[0, 0],
                "stress_total_yy_GPa": total_stress[1, 1],
                "stress_total_zz_GPa": total_stress[2, 2],
                "stress_total_xy_GPa": total_stress[0, 1],
                "stress_total_xz_GPa": total_stress[0, 2],
                "stress_total_yz_GPa": total_stress[1, 2],
                "total_energy_eV": energy_after,
                "energy_before_relax_eV": energy_before,
                "energy_change_eV": energy_after - energy_before,
                "optimization_converged": converged,
                "is_reference_state": False,
                "optimization_status": f"Internal relax: {'SUCCESS' if converged else 'FAILED'}",
            }

        c11_data_all.append(c11_row)
        c12_data_all.append(c12_row)

        logger.debug(
            f"  应变={strain:+.4f}: C11应力={stress_xx:.4f} GPa, C12应力={stress_yy:.4f} GPa, 收敛={converged}"
        )

    # 保存联合数据
    csv_filename = f"c11_c12_combined_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.csv"
    csv_filepath = os.path.join(run_dir, csv_filename)

    # 合并C11和C12数据
    combined_data = c11_data_all + c12_data_all
    df = pd.DataFrame(combined_data)
    df.to_csv(csv_filepath, index=False)

    logger.info(f"💾 C11/C12联合数据已保存: {csv_filename}")
    print(f"  📋 C11/C12联合CSV: {csv_filename}")

    # 分别拟合C11和C12
    # C11拟合
    c11_converged_data = [row for row in c11_data_all if row["optimization_converged"]]
    if len(c11_converged_data) >= 2:
        c11_strains = [row["applied_strain"] for row in c11_converged_data]
        c11_stresses = [row["measured_stress_GPa"] for row in c11_converged_data]
        c11_coeffs = np.polyfit(c11_strains, c11_stresses, 1)
        C11_fitted = c11_coeffs[0]
    else:
        C11_fitted = 0.0

    # C12拟合
    c12_converged_data = [row for row in c12_data_all if row["optimization_converged"]]
    if len(c12_converged_data) >= 2:
        c12_strains = [row["applied_strain"] for row in c12_converged_data]
        c12_stresses = [row["measured_stress_GPa"] for row in c12_converged_data]
        c12_coeffs = np.polyfit(c12_strains, c12_stresses, 1)
        C12_fitted = c12_coeffs[0]
    else:
        C12_fitted = 0.0

    print(
        f"  C11拟合结果 = {C11_fitted:.1f} GPa (文献: 110, 误差: {(C11_fitted / 110 - 1) * 100:+.1f}%)"
    )
    print(
        f"  C12拟合结果 = {C12_fitted:.1f} GPa (文献: 61, 误差: {(C12_fitted / 61 - 1) * 100:+.1f}%)"
    )

    # 生成联合可视化图
    plot_filename = plot_c11_c12_combined_response(
        supercell_size, c11_data_all, c12_data_all, run_dir
    )

    return {
        "C11": C11_fitted,
        "C12": C12_fitted,
        "success": C11_fitted > 0 and C12_fitted > 0,
        "csv_file": csv_filename,
        "plot_file": plot_filename,
        "c11_converged_count": sum(
            1 for row in c11_data_all if row["optimization_converged"]
        ),
        "c12_converged_count": sum(
            1 for row in c12_data_all if row["optimization_converged"]
        ),
        "total_count": len(c11_data_all),
        "csv_data": combined_data,
    }


def plot_c11_c12_combined_response(supercell_size, c11_data, c12_data, run_dir):
    """
    生成C11/C12联合应力-应变响应关系图
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 准备C11数据
    c11_strains = [row["applied_strain"] for row in c11_data]
    c11_stresses = [row["measured_stress_GPa"] for row in c11_data]
    c11_converged_states = [row["optimization_converged"] for row in c11_data]

    # 准备C12数据
    c12_strains = [row["applied_strain"] for row in c12_data]
    c12_stresses = [row["measured_stress_GPa"] for row in c12_data]
    c12_converged_states = [row["optimization_converged"] for row in c12_data]

    # 分别绘制收敛和不收敛的点
    # C11图
    c11_converged_strains = [
        s for s, c in zip(c11_strains, c11_converged_states, strict=False) if c
    ]
    c11_converged_stresses = [
        st for st, c in zip(c11_stresses, c11_converged_states, strict=False) if c
    ]
    c11_failed_strains = [
        s for s, c in zip(c11_strains, c11_converged_states, strict=False) if not c
    ]
    c11_failed_stresses = [
        st for st, c in zip(c11_stresses, c11_converged_states, strict=False) if not c
    ]

    if c11_converged_strains:
        ax1.scatter(
            c11_converged_strains,
            c11_converged_stresses,
            marker="o",
            color="#2E86C1",
            s=80,
            label="C11 (收敛)",
            alpha=0.8,
            edgecolors="black",
        )

    if c11_failed_strains:
        ax1.scatter(
            c11_failed_strains,
            c11_failed_stresses,
            marker="o",
            facecolors="none",
            edgecolors="#2E86C1",
            s=80,
            label="C11 (未收敛)",
            alpha=0.8,
            linewidth=2,
        )

    # C11理论线和拟合线
    literature_C11 = 110.0  # GPa
    strain_range = np.linspace(-0.003, 0.003, 100)
    theory_stress = literature_C11 * strain_range
    ax1.plot(
        strain_range,
        theory_stress,
        "k:",
        linewidth=2,
        alpha=0.7,
        label=f"理论斜率 ({literature_C11} GPa)",
    )

    if len(c11_converged_strains) >= 2:
        coeffs = np.polyfit(c11_converged_strains, c11_converged_stresses, 1)
        fit_strains = np.linspace(
            min(c11_converged_strains), max(c11_converged_strains), 100
        )
        fit_stresses = np.polyval(coeffs, fit_strains)
        ax1.plot(
            fit_strains,
            fit_stresses,
            "--",
            color="#2E86C1",
            alpha=0.7,
            linewidth=2,
            label=f"拟合斜率 ({coeffs[0]:.1f} GPa)",
        )
        C11_fitted = coeffs[0]
    else:
        C11_fitted = 0.0

    ax1.set_xlabel("单轴应变 εxx", fontsize=12)
    ax1.set_ylabel("同轴应力 σxx (GPa)", fontsize=12)
    ax1.set_title(
        "C11: xx应变→xx应力",
        fontsize=13,
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="best")

    # C12图
    c12_converged_strains = [
        s for s, c in zip(c12_strains, c12_converged_states, strict=False) if c
    ]
    c12_converged_stresses = [
        st for st, c in zip(c12_stresses, c12_converged_states, strict=False) if c
    ]
    c12_failed_strains = [
        s for s, c in zip(c12_strains, c12_converged_states, strict=False) if not c
    ]
    c12_failed_stresses = [
        st for st, c in zip(c12_stresses, c12_converged_states, strict=False) if not c
    ]

    if c12_converged_strains:
        ax2.scatter(
            c12_converged_strains,
            c12_converged_stresses,
            marker="s",
            color="#E74C3C",
            s=80,
            label="C12 (收敛)",
            alpha=0.8,
            edgecolors="black",
        )

    if c12_failed_strains:
        ax2.scatter(
            c12_failed_strains,
            c12_failed_stresses,
            marker="s",
            facecolors="none",
            edgecolors="#E74C3C",
            s=80,
            label="C12 (未收敛)",
            alpha=0.8,
            linewidth=2,
        )

    # C12理论线和拟合线
    literature_C12 = 61.0  # GPa
    theory_stress = literature_C12 * strain_range
    ax2.plot(
        strain_range,
        theory_stress,
        "k:",
        linewidth=2,
        alpha=0.7,
        label=f"理论斜率 ({literature_C12} GPa)",
    )

    if len(c12_converged_strains) >= 2:
        coeffs = np.polyfit(c12_converged_strains, c12_converged_stresses, 1)
        fit_strains = np.linspace(
            min(c12_converged_strains), max(c12_converged_strains), 100
        )
        fit_stresses = np.polyval(coeffs, fit_strains)
        ax2.plot(
            fit_strains,
            fit_stresses,
            "--",
            color="#E74C3C",
            alpha=0.7,
            linewidth=2,
            label=f"拟合斜率 ({coeffs[0]:.1f} GPa)",
        )
        C12_fitted = coeffs[0]
    else:
        C12_fitted = 0.0

    ax2.set_xlabel("单轴应变 εxx", fontsize=12)
    ax2.set_ylabel("横向应力 σyy (GPa)", fontsize=12)
    ax2.set_title(
        "C12: xx应变→yy应力",
        fontsize=13,
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="best")

    # C11单独对比图
    convergence_rate_c11 = sum(c11_converged_states) / len(c11_converged_states)

    bar1 = ax3.bar(
        ["C11"],
        [C11_fitted],
        color="#2E86C1",
        alpha=0.3 + 0.7 * convergence_rate_c11,
        edgecolor="black",
        linewidth=1,
        width=0.6,
    )

    # C11文献值参考线
    ax3.axhline(
        y=literature_C11,
        color="#2E86C1",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"文献值 ({literature_C11} GPa)",
    )

    # C11数值标签
    if C11_fitted > 0:
        height = bar1[0].get_height()
        ax3.text(
            bar1[0].get_x() + bar1[0].get_width() / 2.0,
            height + max(height * 0.02, 2),
            f"{C11_fitted:.1f}",
            ha="center",
            va="bottom",
            fontsize=14,
            weight="bold",
        )

        # C11误差
        error = (C11_fitted - literature_C11) / literature_C11 * 100
        ax3.text(
            bar1[0].get_x() + bar1[0].get_width() / 2.0,
            height + max(height * 0.08, 8),
            f"({error:+.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            color="gray",
        )

        # C11收敛率
        ax3.text(
            bar1[0].get_x() + bar1[0].get_width() / 2.0,
            height / 2,
            f"{convergence_rate_c11:.0%}",
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            weight="bold",
        )

    ax3.set_ylabel("弹性常数 (GPa)", fontsize=12)
    ax3.set_title("C11计算结果", fontsize=13)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, max(C11_fitted if C11_fitted > 0 else 0, literature_C11) * 1.3)

    # C12单独对比图
    convergence_rate_c12 = sum(c12_converged_states) / len(c12_converged_states)

    bar2 = ax4.bar(
        ["C12"],
        [C12_fitted],
        color="#E74C3C",
        alpha=0.3 + 0.7 * convergence_rate_c12,
        edgecolor="black",
        linewidth=1,
        width=0.6,
    )

    # C12文献值参考线
    ax4.axhline(
        y=literature_C12,
        color="#E74C3C",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"文献值 ({literature_C12} GPa)",
    )

    # C12数值标签
    if C12_fitted > 0:
        height = bar2[0].get_height()
        ax4.text(
            bar2[0].get_x() + bar2[0].get_width() / 2.0,
            height + max(height * 0.02, 2),
            f"{C12_fitted:.1f}",
            ha="center",
            va="bottom",
            fontsize=14,
            weight="bold",
        )

        # C12误差
        error = (C12_fitted - literature_C12) / literature_C12 * 100
        ax4.text(
            bar2[0].get_x() + bar2[0].get_width() / 2.0,
            height + max(height * 0.08, 8),
            f"({error:+.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            color="gray",
        )

        # C12收敛率
        ax4.text(
            bar2[0].get_x() + bar2[0].get_width() / 2.0,
            height / 2,
            f"{convergence_rate_c12:.0%}",
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            weight="bold",
        )

    ax4.set_ylabel("弹性常数 (GPa)", fontsize=12)
    ax4.set_title("C12计算结果", fontsize=13)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.legend(fontsize=10)
    ax4.set_ylim(0, max(C12_fitted if C12_fitted > 0 else 0, literature_C12) * 1.3)

    plt.suptitle(
        f"C11/C12联合计算 - {supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}系统",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout()

    # 保存图片到运行目录
    filename = f"c11_c12_combined_response_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.png"
    filepath = os.path.join(run_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  📊 C11/C12联合图已保存: {filename}")
    return filename


def apply_volume_strain(cell, strain):
    """应用单轴体积应变"""
    new_cell = cell.copy()
    lattice = new_cell.lattice_vectors.copy()
    lattice[0, 0] *= 1 + strain  # εxx = strain
    new_cell.lattice_vectors = lattice

    # 原子位置也要相应缩放
    positions = new_cell.get_positions()
    positions[:, 0] *= 1 + strain
    new_cell.set_positions(positions)

    return new_cell


def apply_yy_strain(cell, strain):
    """应用yy方向单轴应变用于C12计算"""
    new_cell = cell.copy()
    lattice = new_cell.lattice_vectors.copy()
    lattice[1, 1] *= 1 + strain  # εyy = strain
    new_cell.lattice_vectors = lattice

    # 原子位置也要相应缩放
    positions = new_cell.get_positions()
    positions[:, 1] *= 1 + strain
    new_cell.set_positions(positions)

    return new_cell


def calculate_c12_method(supercell_size, potential, relaxer, run_dir):
    """
    C12计算：施加yy应变，测量xx应力 (Voigt: 2→1)
    """
    logger = logging.getLogger(__name__)

    # 创建系统
    cell = create_aluminum_fcc(supercell_size)

    print("\n计算C12 - yy应变→xx应力")
    print(
        f"系统: {supercell_size[0]}×{supercell_size[1]}×{supercell_size[2]} ({cell.num_atoms}原子)"
    )

    # 基态弛豫 - 使用等比例晶格弛豫（更快且保持对称性）
    base_cell = cell.copy()
    relaxer.uniform_lattice_relax(base_cell, potential)
    base_energy = potential.calculate_energy(base_cell)

    logger.info("C12基态诊断: 系统设置完成")

    # 相同的应变点（与C11一致）
    strain_points = np.array(
        [-0.003, -0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002, 0.003]
    )

    csv_data_all = []

    # 测试yy应变→xx应力 (C12相关)
    for strain in strain_points:
        if strain == 0.0:
            # 基态点 - 完整应力分析
            from thermoelasticsim.elastic.mechanics import StressCalculator

            stress_calc = StressCalculator()
            stress_components = stress_calc.get_all_stress_components(
                base_cell, potential
            )

            # kinetic_stress = stress_components["kinetic"] * EV_TO_GPA  # 暂未使用
            # virial_stress = stress_components["virial"] * EV_TO_GPA    # 暂未使用
            total_stress = stress_components["total"] * EV_TO_GPA
            finite_diff_stress = stress_components["finite_diff"] * EV_TO_GPA

            stress_xx = total_stress[0, 0]  # C12测量xx应力
            converged = True
            energy = base_energy

            csv_row = {
                "method": "C12_cross",
                "strain_direction": "yy",  # 施加yy应变
                "stress_direction": "xx",  # 测量xx应力
                "strain": strain,
                "stress_GPa": stress_xx,
                # 基础应力分量（动能+维里）
                "stress_total_xx_GPa": total_stress[0, 0],
                "stress_total_yy_GPa": total_stress[1, 1],
                "stress_total_zz_GPa": total_stress[2, 2],
                "stress_total_xy_GPa": total_stress[0, 1],
                "stress_total_xz_GPa": total_stress[0, 2],
                "stress_total_yz_GPa": total_stress[1, 2],
                # 晶格应力分量（∂U/∂h）
                "stress_finite_diff_xx_GPa": finite_diff_stress[0, 0],
                "stress_finite_diff_yy_GPa": finite_diff_stress[1, 1],
                "stress_finite_diff_zz_GPa": finite_diff_stress[2, 2],
                "stress_finite_diff_xy_GPa": finite_diff_stress[0, 1],
                "stress_finite_diff_xz_GPa": finite_diff_stress[0, 2],
                "stress_finite_diff_yz_GPa": finite_diff_stress[1, 2],
                # 总应力分量
                "stress_total_xx_GPa": total_stress[0, 0],
                "stress_total_yy_GPa": total_stress[1, 1],
                "stress_total_zz_GPa": total_stress[2, 2],
                "stress_total_xy_GPa": total_stress[0, 1],
                "stress_total_xz_GPa": total_stress[0, 2],
                "stress_total_yz_GPa": total_stress[1, 2],
                "energy_eV": energy,
                "converged": converged,
                "base_state": True,
                "optimization_details": "Base state (fully relaxed)",
            }
        else:
            # yy应变点
            deformed_cell = apply_yy_strain(base_cell, strain)

            # 记录变形前能量
            energy_before = potential.calculate_energy(deformed_cell)

            # 尝试内部弛豫
            converged = relaxer.internal_relax(deformed_cell, potential)

            # 记录变形后状态 - 完整应力分析
            energy_after = potential.calculate_energy(deformed_cell)

            # 获取完整应力分析
            from thermoelasticsim.elastic.mechanics import StressCalculator

            stress_calc = StressCalculator()
            stress_components = stress_calc.get_all_stress_components(
                deformed_cell, potential
            )

            # kinetic_stress = stress_components["kinetic"] * EV_TO_GPA  # 暂未使用
            # virial_stress = stress_components["virial"] * EV_TO_GPA    # 暂未使用
            total_stress = stress_components["total"] * EV_TO_GPA
            finite_diff_stress = stress_components["finite_diff"] * EV_TO_GPA

            stress_xx = total_stress[0, 0]  # C12测量xx应力

            csv_row = {
                "method": "C12_cross",
                "strain_direction": "yy",  # 施加yy应变
                "stress_direction": "xx",  # 测量xx应力
                "strain": strain,
                "stress_GPa": stress_xx,
                # 基础应力分量（动能+维里）
                "stress_total_xx_GPa": total_stress[0, 0],
                "stress_total_yy_GPa": total_stress[1, 1],
                "stress_total_zz_GPa": total_stress[2, 2],
                "stress_total_xy_GPa": total_stress[0, 1],
                "stress_total_xz_GPa": total_stress[0, 2],
                "stress_total_yz_GPa": total_stress[1, 2],
                # 晶格应力分量（∂U/∂h）
                "stress_finite_diff_xx_GPa": finite_diff_stress[0, 0],
                "stress_finite_diff_yy_GPa": finite_diff_stress[1, 1],
                "stress_finite_diff_zz_GPa": finite_diff_stress[2, 2],
                "stress_finite_diff_xy_GPa": finite_diff_stress[0, 1],
                "stress_finite_diff_xz_GPa": finite_diff_stress[0, 2],
                "stress_finite_diff_yz_GPa": finite_diff_stress[1, 2],
                # 总应力分量
                "stress_total_xx_GPa": total_stress[0, 0],
                "stress_total_yy_GPa": total_stress[1, 1],
                "stress_total_zz_GPa": total_stress[2, 2],
                "stress_total_xy_GPa": total_stress[0, 1],
                "stress_total_xz_GPa": total_stress[0, 2],
                "stress_total_yz_GPa": total_stress[1, 2],
                "energy_eV": energy_after,
                "energy_before_relax_eV": energy_before,
                "energy_change_eV": energy_after - energy_before,
                "converged": converged,
                "base_state": False,
                "optimization_details": f"Internal relax: {'SUCCESS' if converged else 'FAILED'}",
            }

        csv_data_all.append(csv_row)
        logger.debug(
            f"  C12应变={strain:+.4f}: xx应力={stress_xx:.4f} GPa, 收敛={converged}"
        )

    # 保存C12详细数据
    csv_filename = (
        f"c12_detailed_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.csv"
    )
    csv_filepath = os.path.join(run_dir, csv_filename)
    df = pd.DataFrame(csv_data_all)
    df.to_csv(csv_filepath, index=False)

    logger.info(f"💾 C12详细数据已保存: {csv_filename}")
    print(f"  📋 C12详细CSV: {csv_filename}")

    # 简单拟合计算C12
    converged_data = [row for row in csv_data_all if row["converged"]]
    if len(converged_data) >= 2:
        strains = [row["strain"] for row in converged_data]
        stresses = [row["stress_GPa"] for row in converged_data]
        coeffs = np.polyfit(strains, stresses, 1)
        C12_fitted = coeffs[0]
    else:
        C12_fitted = 0.0

    print(
        f"  C12拟合结果 = {C12_fitted:.1f} GPa (文献: 61, 误差: {(C12_fitted / 61 - 1) * 100:+.1f}%)"
    )

    # 生成C12应力-应变图
    c12_plot_filename = plot_c12_stress_strain_response(
        supercell_size, csv_data_all, run_dir
    )

    return {
        "C12": C12_fitted,
        "success": C12_fitted != 0.0,
        "csv_file": csv_filename,
        "plot_file": c12_plot_filename,
        "converged_count": sum(1 for row in csv_data_all if row["converged"]),
        "total_count": len(csv_data_all),
        "csv_data": csv_data_all,
    }


def calculate_c11_c12_standard_method(supercell_size, potential, run_dir):
    """
    仿照v3使用标准方法计算C11/C12 - 对比能量变化模式
    """
    logger = logging.getLogger(__name__)

    # 创建系统
    cell = create_aluminum_fcc(supercell_size)

    print("\n计算C11/C12对比 - 标准方法")
    print(
        f"系统: {supercell_size[0]}×{supercell_size[1]}×{supercell_size[2]} ({cell.num_atoms}原子)"
    )

    # 使用ZeroTempDeformationCalculator
    calculator = ZeroTempDeformationCalculator(
        cell=cell,
        potential=potential,
        delta=0.001,  # 应变步长0.1%
        num_steps=6,  # ±6步，总共12个点
        relaxer_params={
            "optimizer_type": "L-BFGS",
            "optimizer_params": {
                "ftol": 1e-7,
                "gtol": 1e-6,
                "maxiter": 2000,
                "maxls": 500,
                "maxfun": 15000,
            },
        },
        supercell_dims=supercell_size,
    )

    logger.info("开始C11/C12标准计算")
    logger.info(
        f"应变范围: ±{calculator.delta * calculator.num_steps * 100:.1f}% ({2 * calculator.num_steps + 1}个点)"
    )

    start_time = time.time()

    try:
        # 执行计算
        C_matrix, r2_score = calculator.calculate()
        calc_time = time.time() - start_time

        # 提取弹性常数
        C11 = C_matrix[0, 0]
        C12 = C_matrix[0, 1]
        C44 = C_matrix[3, 3]

        # 检查对称性
        C_matrix_symmetry = np.max(np.abs(C_matrix - C_matrix.T))

        # 文献值
        lit_C11, lit_C12, lit_C44 = 110.0, 61.0, 33.0

        # 保存结果到CSV
        csv_filename = f"c11_c12_standard_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.csv"
        csv_filepath = os.path.join(run_dir, csv_filename)

        results_data = {
            "component": ["C11", "C12", "C44", "bulk_modulus", "shear_modulus"],
            "value_GPa": [C11, C12, C44, (C11 + 2 * C12) / 3, C44],
            "literature_GPa": [
                lit_C11,
                lit_C12,
                lit_C44,
                (lit_C11 + 2 * lit_C12) / 3,
                lit_C44,
            ],
            "error_percent": [
                (C11 / lit_C11 - 1) * 100,
                (C12 / lit_C12 - 1) * 100,
                (C44 / lit_C44 - 1) * 100,
                ((C11 + 2 * C12) / 3 - (lit_C11 + 2 * lit_C12) / 3)
                / ((lit_C11 + 2 * lit_C12) / 3)
                * 100,
                (C44 / lit_C44 - 1) * 100,
            ],
            "r2_score": [r2_score] * 5,
            "matrix_symmetry": [C_matrix_symmetry] * 5,
            "calculation_time_s": [calc_time] * 5,
        }

        df = pd.DataFrame(results_data)
        df.to_csv(csv_filepath, index=False)

        logger.info(f"C11/C12计算完成: C11={C11:.1f}, C12={C12:.1f}, C44={C44:.1f} GPa")
        logger.info(f"R²={r2_score:.4f}, 对称性={C_matrix_symmetry:.2e}")
        logger.info(f"💾 C11/C12数据已保存: {csv_filename}")

        print(
            f"  C11 = {C11:7.1f} GPa (文献: {lit_C11:.0f}, 误差: {(C11 / lit_C11 - 1) * 100:+.1f}%)"
        )
        print(
            f"  C12 = {C12:7.1f} GPa (文献: {lit_C12:.0f}, 误差: {(C12 / lit_C12 - 1) * 100:+.1f}%)"
        )
        print(
            f"  C44 = {C44:7.1f} GPa (文献: {lit_C44:.0f}, 误差: {(C44 / lit_C44 - 1) * 100:+.1f}%)"
        )
        print(f"  R² = {r2_score:.4f}")
        print(f"  📋 标准方法CSV: {csv_filename}")

        return {
            "C11": C11,
            "C12": C12,
            "C44": C44,
            "r2": r2_score,
            "symmetry": C_matrix_symmetry,
            "time": calc_time,
            "success": True,
            "csv_file": csv_filename,
        }

    except Exception as e:
        logger.error(f"C11/C12计算失败: {e}")
        print(f"  ❌ C11/C12计算失败: {e}")
        return {"success": False, "error": str(e)}


def test_system_size_c44(supercell_size, potential, run_dir, strain_magnitude=0.0001):
    """测试特定尺寸系统的C44计算"""
    nx, ny, nz = supercell_size

    print(f"\n{'=' * 80}")
    print(f"测试 {nx}×{ny}×{nz} 系统 ({nx * ny * nz * 4} 原子) - LAMMPS方法")
    print(f"{'=' * 80}")

    logger = logging.getLogger(__name__)
    logger.info(f"开始测试 {nx}×{ny}×{nz} 系统")

    # 优化器设置（基于成功配置+maxls修复）
    relaxer = StructureRelaxer(
        optimizer_type="L-BFGS",
        optimizer_params={
            # 🔧 关键修复：防止ABNORMAL和提高收敛稳定性
            "ftol": 1e-7,  # 放宽能量收敛以提高稳定性
            "gtol": 1e-10,  # 放宽梯度收敛以提高稳定性
            "maxiter": 10000,  # 减少迭代数避免精度损失
            "maxls": 200,  # 限制线搜索步数防止ABNORMAL
            "maxfun": 50000,  # 限制函数评估次数
            "disp": True,  # 关闭详细输出
        },
        supercell_dims=supercell_size,
    )

    print("配置参数:")
    print(f"  应变幅度: {strain_magnitude * 100:.3f}%")
    print("  基态弛豫: 等比例晶格优化 (uniform_lattice_relax)")
    print(
        f"  收敛条件: ftol={relaxer.optimizer_params['ftol']}, gtol={relaxer.optimizer_params['gtol']} (改进的稳定参数)"
    )
    print(f"  最大迭代: {relaxer.optimizer_params['maxiter']} (防止精度损失)")
    print(f"  线搜索限制: {relaxer.optimizer_params['maxls']} (防止ABNORMAL)")
    print(f"  函数评估: {relaxer.optimizer_params['maxfun']}")

    # 日志记录优化器参数传递
    logger.info("优化器参数传递验证:")
    logger.info(f"  传递的参数: {relaxer.optimizer_params}")
    logger.info(f"  优化器类型: {relaxer.optimizer_type}")

    start_time = time.time()

    try:
        # 1. 高效联合计算C11/C12（一次应变双重收获）
        c11_c12_result = calculate_c11_c12_combined_method(
            supercell_size, potential, relaxer, run_dir
        )

        # 2. 计算C44
        c44_result = calculate_c44_lammps_method(
            supercell_size, strain_magnitude, potential, relaxer, run_dir
        )
        calc_time = time.time() - start_time
        c44_result["time"] = calc_time

        # 3. 合并数据和统计对比
        print("\n📊 收敛率对比:")
        print(
            f"  C11单轴变形(xx→xx): {c11_c12_result['c11_converged_count']}/{c11_c12_result['total_count']} = {c11_c12_result['c11_converged_count'] / c11_c12_result['total_count']:.1%}"
        )
        print(
            f"  C12交叉变形(xx→yy): {c11_c12_result['c12_converged_count']}/{c11_c12_result['total_count']} = {c11_c12_result['c12_converged_count'] / c11_c12_result['total_count']:.1%}"
        )

        total_c44_converged = sum(
            result["converged_count"] for result in c44_result["detailed_results"]
        )
        total_c44_points = sum(
            result["total_count"] for result in c44_result["detailed_results"]
        )
        print(
            f"  C44剪切变形(shear): {total_c44_converged}/{total_c44_points} = {total_c44_converged / total_c44_points:.1%}"
        )

        # 合并CSV数据
        print("\n📋 合并详细数据到统一CSV...")
        combined_csv_filename = f"combined_elastic_data_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.csv"
        combined_csv_path = os.path.join(run_dir, combined_csv_filename)

        # 读取C11/C12联合数据和C44数据
        c11_c12_csv_path = os.path.join(run_dir, c11_c12_result["csv_file"])
        c11_c12_df = pd.read_csv(c11_c12_csv_path)

        c44_csv_path = os.path.join(run_dir, c44_result["csv_file"])
        c44_df = pd.read_csv(c44_csv_path)

        # 合并并保存，统一键名
        combined_df = pd.concat([c11_c12_df, c44_df], ignore_index=True)
        # 确保列名一致性
        if "method" in combined_df.columns:
            combined_df = combined_df.rename(
                columns={
                    "method": "calculation_method",
                    "strain_direction": "applied_strain_direction",
                    "stress_direction": "measured_stress_direction",
                    "strain": "applied_strain",
                    "stress_GPa": "measured_stress_GPa",
                    "energy_eV": "total_energy_eV",
                    "converged": "optimization_converged",
                    "base_state": "is_reference_state",
                    "optimization_details": "optimization_status",
                }
            )
        combined_df.to_csv(combined_csv_path, index=False)

        logger.info(f"💾 合并弹性常数数据已保存: {combined_csv_filename}")
        print(f"  📋 合并弹性常数CSV: {combined_csv_filename}")
        print(f"  📊 C11/C12联合可视化: {c11_c12_result['plot_file']}")
        print(f"  📊 C44可视化: {c44_result['plot_file']}")
        if "trajectory_file" in c44_result:
            print(f"  🗂️ C44轨迹数据: {c44_result['trajectory_file']}")
        if "animation_files" in c44_result and c44_result["animation_files"]:
            animation_files = c44_result["animation_files"]
            if "html" in animation_files:
                print(f"  🎬 交互式动画: {animation_files['html']}")
            if "gif" in animation_files:
                print(f"  📱 轨迹GIF: {animation_files['gif']}")

        results = c44_result

        # 输出结果
        print("\n🎯 弹性常数结果:")
        print(
            f"  C11 = {c11_c12_result['C11']:7.1f} GPa (文献: 110, 误差: {(c11_c12_result['C11'] / 110 - 1) * 100:+.1f}%)"
        )
        print(
            f"  C12 = {c11_c12_result['C12']:7.1f} GPa (文献: 61, 误差: {(c11_c12_result['C12'] / 61 - 1) * 100:+.1f}%)"
        )
        for i, (name, C) in enumerate(
            zip(["C44", "C55", "C66"], results["elastic_constants"], strict=False)
        ):
            error = (C - 33) / 33 * 100
            print(f"  {name} = {C:7.1f} GPa (误差: {error:+.1f}%)")

        print("\n立方对称化:")
        print(f"  平均C44 = {results['C44']:7.1f} GPa")
        print(f"  标准差 = {results['std_dev']:7.1f} GPa")
        print("  文献值 = 33.0 GPa")
        print(f"  总误差 = {results['error_percent']:+.1f}%")

        # 质量指标
        print("\n质量评估:")
        checks = results["quality_checks"]
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {check}: {status}")
        print(f"  成功得分: {results['success_score']:.1%}")

        # 诊断信息
        print("\n系统诊断:")
        print(f"  基态应力: {results['base_stress_magnitude']:.4f} GPa")
        print(f"  晶格对称性: {results['asymmetry']:.2e} Å")
        print(f"  计算时间: {calc_time:.1f} 秒")

        if results["success_score"] >= 0.8:
            print("  🎉 配置优秀！")
        elif results["success_score"] >= 0.6:
            print("  ✅ 配置良好")
        else:
            print("  ⚠️ 需要改进")

        logger.info(
            f"测试完成: C44={results['C44']:.1f} GPa, 误差={results['error_percent']:+.1f}%"
        )

        # 将最终结果写入日志
        logger.info("=" * 60)
        logger.info(
            f"最终弹性常数结果 - {supercell_size[0]}×{supercell_size[1]}×{supercell_size[2]} 系统"
        )
        logger.info(f"原子数量: {results['atoms']}")
        logger.info(f"应变幅度: {strain_magnitude * 100:.3f}%")
        for i, (name, C) in enumerate(
            zip(["C44", "C55", "C66"], results["elastic_constants"], strict=False)
        ):
            error = (C - 33) / 33 * 100
            logger.info(f"  {name} = {C:7.1f} GPa (误差: {error:+5.1f}%)")
        logger.info("立方对称化结果:")
        logger.info(f"  平均C44 = {results['C44']:7.1f} GPa")
        logger.info(f"  标准差   = {results['std_dev']:7.1f} GPa")
        logger.info("  文献值   = 33.0 GPa")
        logger.info(f"  总误差   = {results['error_percent']:+6.1f}%")
        logger.info(f"质量评估得分: {results['success_score']:.1%}")
        logger.info(f"计算时间: {results['time']:.1f} 秒")

        # 对比C11/C12联合计算结果
        if c11_c12_result.get("success"):
            logger.info("C11/C12联合计算结果:")
            logger.info(f"  C11 = {c11_c12_result['C11']:7.1f} GPa (文献: 110 GPa)")
            logger.info(f"  C12 = {c11_c12_result['C12']:7.1f} GPa (文献: 61 GPa)")
            logger.info(
                f"  C11收敛率 = {c11_c12_result['c11_converged_count']}/{c11_c12_result['total_count']} ({c11_c12_result['c11_converged_count'] / c11_c12_result['total_count']:.1%})"
            )
            logger.info(
                f"  C12收敛率 = {c11_c12_result['c12_converged_count']}/{c11_c12_result['total_count']} ({c11_c12_result['c12_converged_count'] / c11_c12_result['total_count']:.1%})"
            )
            logger.info(f"  联合可视化图: {c11_c12_result['plot_file']}")

        logger.info("收敛率对比:")
        logger.info(
            f"  C11/C12联合: {(c11_c12_result['c11_converged_count'] + c11_c12_result['c12_converged_count']) / (c11_c12_result['total_count'] * 2):.1%}"
        )
        logger.info(f"  C44剪切: {total_c44_converged / total_c44_points:.1%}")
        logger.info(f"合并数据文件: {combined_csv_filename}")
        logger.info(
            f"可视化图表: C11/C12联合({c11_c12_result.get('plot_file', 'N/A')}), C44({c44_result['plot_file']})"
        )

        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"计算失败: {e}")
        print(f"\n❌ 计算失败: {e}")
        return {
            "atoms": nx * ny * nz * 4,
            "success": False,
            "error": str(e),
            "time": time.time() - start_time,
        }


def main():
    # 设置日志
    log_file, run_dir = setup_logging("c44_final_v7")

    print("=" * 80)
    print("C44弹性常数计算 - 最终版本 v7")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志目录: {run_dir}")
    print("=" * 80)

    print("\n🎯 版本特性:")
    print("  ✅ LAMMPS风格盒子剪切方法（关键突破）")
    print("  ✅ 基于18.4%误差成功配置优化")
    print("  ✅ 完整系统尺寸比较和诊断")
    print("  ✅ 立方对称化和质量评估")
    print("  🚀 等比例晶格基态优化（5700x速度提升+完美对称性）")

    logger = logging.getLogger(__name__)
    logger.info("开始C44计算分析")

    # 初始化势能
    print("\n初始化EAM Al1势能...")
    potential = EAMAl1Potential(cutoff=6.5)

    # 测试配置（基于成功经验）
    test_configs = [
        {"size": (2, 2, 2), "strain": 0.001, "desc": "快速验证"},  # 32原子，0.1%
        {
            "size": (3, 3, 3),
            "strain": 0.0001,
            "desc": "最优配置",
        },  # 108原子，0.01%（成功配置）
        {"size": (4, 4, 4), "strain": 0.0001, "desc": "高精度"},  # 256原子，0.01%
    ]

    results = {}
    total_time = 0

    for i, config in enumerate(test_configs):
        print(f"\n进度: [{i + 1}/{len(test_configs)}] - {config['desc']}")
        if total_time > 0:
            print(f"已用时: {total_time / 60:.1f} 分钟")

        start = time.time()
        results[config["size"]] = test_system_size_c44(
            config["size"], potential, run_dir, config["strain"]
        )
        elapsed = time.time() - start
        total_time += elapsed

        # 立即检查是否成功
        if results[config["size"]].get("success"):
            error = results[config["size"]]["error_percent"]
            if abs(error) < 50:  # 可接受误差
                print(f"✅ 配置成功！误差: {error:+.1f}%")
            else:
                print(f"⚠️ 误差较大: {error:+.1f}%")

        # 为避免过度计算，如果找到好结果可提前停止
        if (
            results[config["size"]].get("success")
            and abs(results[config["size"]].get("error_percent", 100)) < 30
        ):
            print("\n✨ 发现良好配置，可选择继续或停止测试")

    # 结果汇总
    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)

    print(
        f"\n{'系统':<8} {'原子':<6} {'C44(GPa)':<8} {'误差(%)':<8} {'得分':<6} {'时间(s)':<8}"
    )
    print("-" * 60)

    best_result = None
    best_error = float("inf")

    for size, result in results.items():
        if result.get("success"):
            error = result["error_percent"]
            if abs(error) < abs(best_error):
                best_error = error
                best_result = (size, result)

            score_str = f"{result['success_score']:.1%}"
            print(
                f"{size[0]}×{size[1]}×{size[2]:<8} {result['atoms']:<6} "
                f"{result['C44']:<8.1f} {error:<8.1f} {score_str:<6} {result['time']:<8.1f}"
            )
        else:
            print(
                f"{size[0]}×{size[1]}×{size[2]:<8} {result['atoms']:<6} {'失败':<8} {'-':<8} {'-':<6} {result['time']:<8.1f}"
            )

    # 最佳配置推荐
    if best_result:
        size, result = best_result
        print("\n🏆 最佳配置推荐:")
        print(f"  系统尺寸: {size[0]}×{size[1]}×{size[2]} ({result['atoms']}原子)")
        print(f"  C44结果: {result['C44']:.1f} GPa")
        print(f"  误差: {result['error_percent']:+.1f}%")
        print(f"  质量得分: {result['success_score']:.1%}")

        if abs(result["error_percent"]) < 25:
            print("  🎉 建议应用到主代码！")
        elif abs(result["error_percent"]) < 50:
            print("  ✅ 配置良好，可考虑进一步优化")
        else:
            print("  ⚠️ 仍需改进")

    print(f"\n总计算时间: {total_time / 60:.1f} 分钟")
    print(f"详细日志目录: {run_dir}")

    # 将汇总结果也写入日志
    logger.info("C44分析汇总结果")
    logger.info("=" * 80)
    for size, result in results.items():
        if result.get("success"):
            error = result["error_percent"]
            logger.info(
                f"{size[0]}×{size[1]}×{size[2]} 系统: C44={result['C44']:.1f} GPa, 误差={error:+.1f}%, 得分={result['success_score']:.1%}, 时间={result['time']:.1f}s"
            )
        else:
            logger.info(
                f"{size[0]}×{size[1]}×{size[2]} 系统: 计算失败, 时间={result['time']:.1f}s, 错误={result.get('error', '未知')}"
            )

    if best_result:
        size, result = best_result
        logger.info(
            f"最佳配置: {size[0]}×{size[1]}×{size[2]} 系统, C44={result['C44']:.1f} GPa, 误差={result['error_percent']:+.1f}%"
        )

    logger.info("C44分析完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
