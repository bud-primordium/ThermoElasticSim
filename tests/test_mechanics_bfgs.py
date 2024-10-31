import logging
import numpy as np
import os
from datetime import datetime


def test_elastic_constants_fcc_aluminum_bfgs(
    large_fcc_cell, lj_potential_with_neighbor_list_fcc
):
    """
    使用 BFGS 优化器测试 FCC 铝的弹性常数计算
    """
    from python.zeroelasticity import ZeroKElasticConstantsCalculator

    logger = logging.getLogger(__name__)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./output", f"test_mechanics_bfgs_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 记录初始状态
    logger.info(f"Test starting with {large_fcc_cell.num_atoms} atoms")
    logger.info(f"Initial cell volume: {large_fcc_cell.volume:.3f} Å³")
    logger.info(
        f"Initial lattice parameters: {np.diag(large_fcc_cell.lattice_vectors)}"
    )

    # 分析初始原子分布
    initial_positions = large_fcc_cell.get_positions()
    logger.info("Analyzing initial atomic distribution...")

    # 计算并记录初始最近邻距离
    def analyze_atomic_distances(cell, stage=""):
        positions = cell.get_positions()
        min_dist = float("inf")
        max_dist = 0.0
        min_pair = max_pair = None

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                rij = positions[j] - positions[i]
                if cell.pbc_enabled:
                    rij = cell.minimum_image(rij)
                dist = np.linalg.norm(rij)

                if dist < min_dist:
                    min_dist = dist
                    min_pair = (i, j)
                if dist > max_dist:
                    max_dist = dist
                    max_pair = (i, j)

        logger.info(f"{stage} Analysis:")
        logger.info(f"  Minimum distance: {min_dist:.3f} Å between atoms {min_pair}")
        logger.info(f"  Maximum distance: {max_dist:.3f} Å between atoms {max_pair}")
        return min_dist, max_dist

    initial_min_dist, initial_max_dist = analyze_atomic_distances(
        large_fcc_cell, "Initial"
    )

    # 分析邻居列表
    neighbor_list = lj_potential_with_neighbor_list_fcc.neighbor_list
    neighbor_counts = [len(neighbors) for neighbors in neighbor_list.neighbor_list]
    logger.info("Neighbor list analysis:")
    logger.info(f"  Average neighbors per atom: {np.mean(neighbor_counts):.1f}")
    logger.info(f"  Min neighbors: {min(neighbor_counts)}")
    logger.info(f"  Max neighbors: {max(neighbor_counts)}")

    # 创建弹性常数计算器
    calculator = ZeroKElasticConstantsCalculator(
        cell=large_fcc_cell,
        potential=lj_potential_with_neighbor_list_fcc,
        delta=0.001,  # 0.1% 应变
        num_steps=5,
        optimizer_type="BFGS",  # 使用 BFGS 优化器
        save_path=output_dir,  # 使用时间戳目录
    )

    # 计算弹性常数
    C = calculator.calculate_elastic_constants()

    # 分析最终状态
    final_min_dist, final_max_dist = analyze_atomic_distances(large_fcc_cell, "Final")

    # 检查原子间距是否合理
    sigma = lj_potential_with_neighbor_list_fcc.parameters["sigma"]
    min_allowed_dist = 0.8 * sigma  # 通常 LJ 势的硬核排斥区
    assert (
        final_min_dist > min_allowed_dist
    ), f"Atoms too close: {final_min_dist:.3f} Å < {min_allowed_dist:.3f} Å"

    # 检查基本对称性
    np.testing.assert_almost_equal(C[0, 0], C[1, 1], decimal=1)
    np.testing.assert_almost_equal(C[0, 0], C[2, 2], decimal=1)
    np.testing.assert_almost_equal(C[3, 3], C[4, 4], decimal=1)
    np.testing.assert_almost_equal(C[3, 3], C[5, 5], decimal=1)

    # 检查数值范围
    logger.info("\nElastic constants (GPa):")
    logger.info(f"C11 = {C[0,0]:.1f}")
    logger.info(f"C12 = {C[0,1]:.1f}")
    logger.info(f"C44 = {C[3,3]:.1f}")

    assert 50 < C[0, 0] < 200, f"C11 = {C[0,0]:.1f} GPa is out of reasonable range"
    assert 20 < C[0, 1] < 100, f"C12 = {C[0,1]:.1f} GPa is out of reasonable range"
    assert 10 < C[3, 3] < 50, f"C44 = {C[3,3]:.1f} GPa is out of reasonable range"
