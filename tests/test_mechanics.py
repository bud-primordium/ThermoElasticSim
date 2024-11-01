# tests/test_mechanics.py

import numpy as np
import logging
from python.zeroelasticity import ZeroKElasticConstantsSolver
from python.mechanics import StressCalculatorLJ, StressCalculatorEAM
from python.potentials import EAMAl1Potential

# 配置日志
logger = logging.getLogger(__name__)


def test_stress_calculation(two_atom_cell, lj_potential_with_neighbor_list):
    """
    测试应力计算器的功能。
    """
    stress_calculator = StressCalculatorLJ()

    # 计算力
    lj_potential_with_neighbor_list.calculate_forces(two_atom_cell)

    # 计算应力
    stress_tensor = stress_calculator.compute_stress(
        two_atom_cell, lj_potential_with_neighbor_list
    )

    # 计算预期的应力张量，假设 epsilon 和 sigma 已定义
    epsilon = 0.0103  # 根据实际情况替换
    sigma = 2.55  # 单位 Å
    force_magnitude = 24 * epsilon / sigma  # 相互作用力大小

    # 晶胞体积
    cell_volume = np.linalg.det(two_atom_cell.lattice_vectors)  # 计算晶胞的体积
    print(f"Cell volume: {cell_volume}")

    # 应力张量的 x 分量（正应力）
    expected_stress_xx = +force_magnitude * sigma / cell_volume

    # 设置预期应力张量，其他分量为零
    expected_stress = np.array(
        [[expected_stress_xx, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    # 检查应力张量是否与预期值相符
    np.testing.assert_array_almost_equal(stress_tensor, expected_stress, decimal=6)


def test_elastic_constants_solver():
    """
    测试弹性常数求解器的功能。
    """
    strains = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01]),
    ]
    stresses = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([69.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 69.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 69.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 23.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 23.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 23.0]),
    ]

    # 创建 ZeroKElasticConstantsSolver 实例
    solver = ZeroKElasticConstantsSolver()

    # 使用 solver.solve 方法计算弹性常数矩阵
    C = solver.solve(strains, stresses)

    # 检查 C 是否为 6x6 矩阵
    assert C.shape == (6, 6), "Elastic constants matrix shape mismatch."
    # 预期弹性常数矩阵
    expected_C = np.diag([6900.0, 6900.0, 6900.0, 2300.0, 2300.0, 2300.0])
    # 检查弹性常数矩阵是否接近预期值
    np.testing.assert_array_almost_equal(C, expected_C, decimal=2)


def test_eam_stress_calculation(two_atom_cell):
    """
    测试EAM应力计算器的功能。

    使用简单的两原子系统验证应力计算的基本功能。
    """
    logger.info("开始测试EAM应力计算器...")

    # 创建EAM势和应力计算器实例
    eam_potential = EAMAl1Potential(cutoff=6.5)
    stress_calculator = StressCalculatorEAM()
    logger.debug(f"已创建EAM势(cutoff={eam_potential.cutoff}Å)和应力计算器")

    # 计算力
    eam_potential.calculate_forces(two_atom_cell)
    forces = two_atom_cell.get_forces()
    logger.debug(f"计算的原子力:\n{forces}")

    # 计算应力
    stress_tensor = stress_calculator.compute_stress(two_atom_cell, eam_potential)
    logger.info(f"计算的应力张量 (eV/Å³):\n{stress_tensor}")

    # 基本验证
    assert stress_tensor.shape == (3, 3), "应力张量维度应该是3x3"
    assert np.allclose(stress_tensor, stress_tensor.T), "应力张量应该是对称的"

    # 检查应力分量的基本物理特性
    cell_volume = np.linalg.det(two_atom_cell.lattice_vectors)
    logger.debug(f"晶胞体积: {cell_volume:.3f} Å³")

    force = two_atom_cell.get_forces()[0]
    position = two_atom_cell.get_positions()[0]
    logger.debug(f"第一个原子位置: {position}")
    logger.debug(f"第一个原子受力: {force}")

    # 计算并记录von Mises应力
    s = stress_tensor
    von_mises = np.sqrt(
        0.5
        * (
            (s[0, 0] - s[1, 1]) ** 2
            + (s[1, 1] - s[2, 2]) ** 2
            + (s[2, 2] - s[0, 0]) ** 2
            + 6.0 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[2, 0] ** 2)
        )
    )
    logger.info(f"von Mises应力: {von_mises:.6f} eV/Å³")

    # 记录主应力
    eigenvalues = np.linalg.eigvals(stress_tensor)
    logger.info(f"主应力: {eigenvalues}")

    logger.info("EAM应力计算测试完成")


def log_stress_analysis(stress_tensor, potential_name):
    """分析并记录应力张量的详细信息"""
    logger.info(f"\n{potential_name}应力分析:")

    # 计算主应力
    eigenvals, eigenvecs = np.linalg.eigh(stress_tensor)
    logger.info(f"主应力值: {eigenvals}")
    logger.info(f"主应力方向:\n{eigenvecs}")

    # 计算静水压力 (应力张量的迹除以3)
    hydrostatic_pressure = np.trace(stress_tensor) / 3
    logger.info(f"静水压力: {hydrostatic_pressure:.6f} eV/Å³")

    # 计算偏应力张量
    deviatoric_stress = stress_tensor - hydrostatic_pressure * np.eye(3)
    logger.info(f"偏应力张量:\n{deviatoric_stress}")

    # 计算第二应力不变量 J2
    j2 = np.sum(deviatoric_stress * deviatoric_stress) / 2
    logger.info(f"第二应力不变量 J2: {j2:.6f} (eV/Å³)²")


def test_stress_comparison(two_atom_cell, lj_potential_with_neighbor_list):
    """
    比较不同势函数计算的应力张量。
    """
    logger.info("开始比较LJ势和EAM势的应力计算...")

    # LJ势的应力计算
    logger.debug("计算LJ势的应力...")
    lj_calculator = StressCalculatorLJ()
    lj_potential_with_neighbor_list.calculate_forces(two_atom_cell)
    lj_stress = lj_calculator.compute_stress(
        two_atom_cell, lj_potential_with_neighbor_list
    )
    logger.info(f"LJ势应力张量 (eV/Å³):\n{lj_stress}")

    # EAM势的应力计算
    logger.debug("计算EAM势的应力...")
    eam_potential = EAMAl1Potential(cutoff=6.5)
    eam_calculator = StressCalculatorEAM()
    eam_potential.calculate_forces(two_atom_cell)
    eam_stress = eam_calculator.compute_stress(two_atom_cell, eam_potential)
    logger.info(f"EAM势应力张量 (eV/Å³):\n{eam_stress}")

    # 计算应力差异
    stress_diff = np.abs(lj_stress - eam_stress)
    logger.info(f"应力差异张量 (eV/Å³):\n{stress_diff}")
    logger.info(f"最大应力差异: {np.max(stress_diff):.6f} eV/Å³")

    # 计算两种方法的von Mises应力
    def calculate_von_mises(s):
        return np.sqrt(
            0.5
            * (
                (s[0, 0] - s[1, 1]) ** 2
                + (s[1, 1] - s[2, 2]) ** 2
                + (s[2, 2] - s[0, 0]) ** 2
                + 6.0 * (s[0, 1] ** 2 + s[1, 2] ** 2 + s[2, 0] ** 2)
            )
        )

    lj_von_mises = calculate_von_mises(lj_stress)
    eam_von_mises = calculate_von_mises(eam_stress)
    logger.info(f"LJ势von Mises应力: {lj_von_mises:.6f} eV/Å³")
    logger.info(f"EAM势von Mises应力: {eam_von_mises:.6f} eV/Å³")

    # 对两种方法的特征值进行比较
    lj_eigenvals = np.sort(np.real(np.linalg.eigvals(lj_stress)))
    eam_eigenvals = np.sort(np.real(np.linalg.eigvals(eam_stress)))
    logger.info(f"LJ势主应力: {lj_eigenvals}")
    logger.info(f"EAM势主应力: {eam_eigenvals}")

    logger.info("应力计算比较测试完成")

    # 添加详细的应力分析
    log_stress_analysis(lj_stress, "LJ势")
    log_stress_analysis(eam_stress, "EAM势")

    # 分析应力差异的性质
    logger.info("\n应力差异分析:")
    logger.info(f"平均应力差异: {np.mean(np.abs(lj_stress - eam_stress)):.6f} eV/Å³")
    logger.info(f"标准差: {np.std(lj_stress - eam_stress):.6f} eV/Å³")

    # 对比静水压力分量与剪切分量的差异
    lj_pressure = np.trace(lj_stress) / 3
    eam_pressure = np.trace(eam_stress) / 3
    logger.info(f"静水压力差异: {lj_pressure - eam_pressure:.6f} eV/Å³")
