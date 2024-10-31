# tests/test_mechanics.py

import numpy as np
import logging
from python.zeroelasticity import ZeroKElasticConstantsSolver
from python.mechanics import StressCalculatorLJ


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


def test_elastic_constants_fcc_aluminum(
    large_fcc_cell, lj_potential_with_neighbor_list_fcc
):
    """
    测试FCC铝的弹性常数计算
    """
    from python.zeroelasticity import ZeroKElasticConstantsCalculator

    # 添加一些调试信息
    logger = logging.getLogger(__name__)
    logger.debug(f"Number of atoms in cell: {large_fcc_cell.num_atoms}")
    logger.debug(f"Cell volume: {large_fcc_cell.volume}")
    logger.debug(f"Neighbor list cutoff: {lj_potential_with_neighbor_list_fcc.cutoff}")

    # 创建弹性常数计算器
    calculator = ZeroKElasticConstantsCalculator(
        cell=large_fcc_cell,
        potential=lj_potential_with_neighbor_list_fcc,
        delta=0.001,  # 0.1% 应变
        num_steps=5,
    )

    # 计算弹性常数
    C = calculator.calculate_elastic_constants()

    # 输出计算结果
    logger.info(f"Calculated elastic constants:\n{C}")

    # FCC铝的实验弹性常数 (GPa)
    C11_exp = 108.0
    C12_exp = 62.0
    C44_exp = 28.0

    # 检查基本的对称性
    np.testing.assert_almost_equal(C[0, 0], C[1, 1], decimal=1)
    np.testing.assert_almost_equal(C[0, 0], C[2, 2], decimal=1)
    np.testing.assert_almost_equal(C[3, 3], C[4, 4], decimal=1)
    np.testing.assert_almost_equal(C[3, 3], C[5, 5], decimal=1)

    # 检查数值范围的合理性
    assert 50 < C[0, 0] < 200, f"C11 = {C[0,0]} GPa is out of reasonable range"
    assert 20 < C[0, 1] < 100, f"C12 = {C[0,1]} GPa is out of reasonable range"
    assert 10 < C[3, 3] < 50, f"C44 = {C[3,3]} GPa is out of reasonable range"
