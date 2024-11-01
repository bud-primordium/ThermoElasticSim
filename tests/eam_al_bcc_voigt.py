# eam_al_bcc_voigt.py
import logging
import numpy as np
import os
from datetime import datetime
from python.structure import Atom, Cell
from python.potentials import EAMAl1Potential
from python.zeroelasticity import ZeroKElasticConstantsCalculator


def test_elastic_constants_bcc_aluminum_eam():
    """
    使用ZeroKElasticConstantsCalculator计算BCC铝的弹性常数
    """
    logger = logging.getLogger(__name__)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./output", f"test_mechanics_eam_bcc_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 创建基本BCC晶胞
    a0 = 4.05  # 晶格常数，单位：Å
    lattice_vectors = np.array([[a0, 0.0, 0.0], [0.0, a0, 0.0], [0.0, 0.0, a0]])

    # BCC原子位置（分数坐标）
    positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]  # 顶角原子  # 体心原子

    # 创建原子列表，转换到笛卡尔坐标
    atoms = []
    for i, frac_pos in enumerate(positions):
        cart_pos = np.dot(frac_pos, lattice_vectors)
        atoms.append(Atom(id=i, symbol="Al", mass_amu=26.98, position=cart_pos))

    # 创建基本晶胞并构建2x2x2超胞
    base_cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    supercell = base_cell.build_supercell((2, 2, 2))

    logger.info(f"Created BCC supercell with {supercell.num_atoms} atoms")
    logger.info(f"Cell dimensions: {np.diag(supercell.lattice_vectors)} Å")

    # 创建EAM势
    eam_potential = EAMAl1Potential()

    # 创建弹性常数计算器
    calculator = ZeroKElasticConstantsCalculator(
        cell=supercell,
        potential=eam_potential,
        delta=0.001,  # 0.1% 应变
        num_steps=7,  # 7个应变点
        optimizer_type="LBFGS",
        optimizer_params={
            "ftol": 1e-8,  # 严格的能量收敛
            "gtol": 1e-5,  # 较严格的力收敛
            "maxls": 100,  # 较大的线搜索步数
            "maxiter": 10000,  # 足够大的迭代次数
        },
        save_path=output_dir,
    )

    # 计算弹性常数
    C = calculator.calculate_elastic_constants()

    # 检查立方晶体的基本对称性
    np.testing.assert_almost_equal(C[0, 0], C[1, 1], decimal=1)  # C11 = C22
    np.testing.assert_almost_equal(C[0, 0], C[2, 2], decimal=1)  # C11 = C33
    np.testing.assert_almost_equal(C[3, 3], C[4, 4], decimal=1)  # C44 = C55
    np.testing.assert_almost_equal(C[3, 3], C[5, 5], decimal=1)  # C44 = C66

    # 输出弹性常数
    logger.info("\nElastic constants (GPa):")
    logger.info(f"C11 = {C[0,0]:.1f}")
    logger.info(f"C12 = {C[0,1]:.1f}")
    logger.info(f"C44 = {C[3,3]:.1f}")

    # 计算各向异性因子 A = 2C44/(C11-C12)
    A = 2 * C[3, 3] / (C[0, 0] - C[0, 1])
    logger.info(f"Anisotropy factor: {A:.3f}")

    # 计算体积模量 B = (C11 + 2C12)/3
    B = (C[0, 0] + 2 * C[0, 1]) / 3
    logger.info(f"Bulk modulus: {B:.1f} GPa")

    # 力学稳定性检查
    logger.info("\n力学稳定性检查:")
    logger.info(f"C11 > 0: {C[0,0] > 0}")
    logger.info(f"C44 > 0: {C[3,3] > 0}")
    logger.info(f"C11 > |C12|: {C[0,0] > abs(C[0,1])}")
    logger.info(f"C11 + 2C12 > 0: {C[0,0] + 2*C[0,1] > 0}")

    return C
