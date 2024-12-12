import logging
import numpy as np
import os
from datetime import datetime
from python.structure import Atom, Cell
from python.potentials import EAMAl1Potential
from python.zeroelasticity import ZeroKElasticConstantsCalculator


def test_elastic_constants_fcc_aluminum_eam():
    """
    使用 EAM Al1 势和 LBFGS 优化器测试 FCC 铝的弹性常数计算
    """
    logger = logging.getLogger(__name__)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./output", f"test_mechanics_eam_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 创建基本FCC铝晶胞
    a0 = 4.046  # 铝的晶格常数，单位：Å
    lattice_vectors = np.array([[a0, 0.0, 0.0], [0.0, a0, 0.0], [0.0, 0.0, a0]])

    # FCC基胞原子位置（分数坐标）
    fcc_positions = np.array(
        [
            [0.0, 0.0, 0.0],  # 顶角
            [0.5, 0.5, 0.0],  # 面心
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )

    # 创建原子列表
    atoms = []
    for i, pos in enumerate(fcc_positions):
        # 转换到笛卡尔坐标
        cart_pos = np.dot(pos, lattice_vectors)
        # 铝原子质量为26.98 amu
        atoms.append(Atom(id=i, symbol="Al", mass_amu=26.98, position=cart_pos))

    # 创建基本晶胞
    base_cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)

    # 创建4x4x4超胞，总共256个原子
    supercell = base_cell.build_supercell((4, 4, 4))

    logger.info(f"Created supercell with {supercell.num_atoms} atoms")
    logger.info(f"Initial cell volume: {supercell.volume:.3f} Å³")
    logger.info(f"Initial lattice parameters: {np.diag(supercell.lattice_vectors)}")

    # 分析初始原子分布
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

    initial_min_dist, initial_max_dist = analyze_atomic_distances(supercell, "Initial")

    # 创建EAM势
    eam_potential = EAMAl1Potential()

    # 创建弹性常数计算器，使用较小的应变和更多的步数以提高精度
    calculator = ZeroKElasticConstantsCalculator(
        cell=supercell,
        potential=eam_potential,
        delta=0.001,  # 0.1% 应变
        num_steps=7,  # 增加步数以获得更好的拟合
        optimizer_type="LBFGS",  # 使用 LBFGS 优化器
        optimizer_params={"tol": 1e-8, "maxiter": 1000},  # 更严格的收敛标准
        save_path=output_dir,
    )

    # 计算弹性常数
    C = calculator.calculate_elastic_constants()

    # 分析最终状态
    final_min_dist, final_max_dist = analyze_atomic_distances(supercell, "Final")

    # 检查原子间距是否合理（对于Al，最近邻距离应该在2.7-2.9Å左右）
    min_allowed_dist = 2.0  # Al的硬核半径约为2.0Å
    assert (
        final_min_dist > min_allowed_dist
    ), f"Atoms too close: {final_min_dist:.3f} Å < {min_allowed_dist:.3f} Å"

    # 检查立方晶体的基本对称性
    np.testing.assert_almost_equal(C[0, 0], C[1, 1], decimal=1)  # C11 = C22
    np.testing.assert_almost_equal(C[0, 0], C[2, 2], decimal=1)  # C11 = C33
    np.testing.assert_almost_equal(C[3, 3], C[4, 4], decimal=1)  # C44 = C55
    np.testing.assert_almost_equal(C[3, 3], C[5, 5], decimal=1)  # C44 = C66

    # 检查数值范围（实验值参考）
    # C11 ≈ 108 GPa
    # C12 ≈ 62 GPa
    # C44 ≈ 28 GPa
    logger.info("\nElastic constants (GPa):")
    logger.info(f"C11 = {C[0,0]:.1f}")
    logger.info(f"C12 = {C[0,1]:.1f}")
    logger.info(f"C44 = {C[3,3]:.1f}")

    assert 80 < C[0, 0] < 140, f"C11 = {C[0,0]:.1f} GPa is out of reasonable range"
    assert 40 < C[0, 1] < 80, f"C12 = {C[0,1]:.1f} GPa is out of reasonable range"
    assert 20 < C[3, 3] < 40, f"C44 = {C[3,3]:.1f} GPa is out of reasonable range"

    # 计算各向异性因子 A = 2C44/(C11-C12)
    A = 2 * C[3, 3] / (C[0, 0] - C[0, 1])
    logger.info(f"Anisotropy factor: {A:.3f}")
    # 铝的各向异性因子应接近1.2
    assert 1.0 < A < 1.4, f"Anisotropy factor {A:.3f} is out of reasonable range"

    # 计算体积模量 B = (C11 + 2C12)/3
    B = (C[0, 0] + 2 * C[0, 1]) / 3
    logger.info(f"Bulk modulus: {B:.1f} GPa")
    # 铝的体积模量应在70-80 GPa范围内
    assert 60 < B < 90, f"Bulk modulus {B:.1f} GPa is out of reasonable range"

    return C
