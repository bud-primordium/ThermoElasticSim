# eam_al_fcc_voigt.py
import logging
import numpy as np
import os
from datetime import datetime
from python.structure import Atom, Cell
from python.potentials import EAMAl1Potential
from python.zeroelasticity import ZeroKElasticConstantsCalculator


def create_fcc_cell(lattice_constant):
    """
    创建基本的 FCC 单胞（4个原子）

    Parameters
    ----------
    lattice_constant : float
        晶格常数，单位：Å

    Returns
    -------
    Cell
        FCC 晶胞对象
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating FCC cell.")

    # 创建晶格向量
    lattice_vectors = np.array(
        [
            [lattice_constant, 0.0, 0.0],
            [0.0, lattice_constant, 0.0],
            [0.0, 0.0, lattice_constant],
        ]
    )

    # 基本 FCC 的分数坐标
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 角顶点
            [0.0, 0.5, 0.5],  # 面心
            [0.5, 0.0, 0.5],  # 面心
            [0.5, 0.5, 0.0],  # 面心
        ]
    )

    # 将分数坐标转换为笛卡尔坐标
    cart_coords = np.dot(frac_coords, lattice_vectors)

    # 创建原子列表
    atoms = []
    for i, pos in enumerate(cart_coords):
        atoms.append(Atom(id=i, symbol="Al", mass_amu=26.98, position=pos))

    # 创建晶胞
    logger.info("FCC cell created with 4 atoms.")
    return Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)


def create_supercell(cell, repetition):
    """
    使用 Cell 类的 build_supercell 方法创建超胞

    Parameters
    ----------
    cell : Cell
        基本晶胞对象
    repetition : tuple of int
        每个晶格方向上的重复次数

    Returns
    -------
    Cell
        超胞对象
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building supercell with repetition factors: {repetition}.")
    supercell = cell.build_supercell(repetition)
    logger.info(f"Supercell created with {supercell.num_atoms} atoms.")
    return supercell


def test_elastic_constants_fcc_aluminum_eam():
    """
    使用 ZeroKElasticConstantsCalculator 计算 FCC 铝的弹性常数
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting elastic constants calculation for FCC Aluminum using EAM potential."
    )

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./output", f"test_mechanics_eam_fcc_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory created at: {output_dir}")

    # 创建基本 FCC 晶胞
    a0 = 4.05  # 晶格常数，单位：Å
    base_cell = create_fcc_cell(lattice_constant=a0)

    # 构建 超胞
    supercell = create_supercell(base_cell, repetition=(4, 4, 4))

    logger.info(f"Created FCC supercell with {supercell.num_atoms} atoms.")
    logger.info(f"Cell dimensions:\n{supercell.lattice_vectors} Å")

    # 创建 EAM 势
    eam_potential = EAMAl1Potential()
    logger.info("EAM potential created.")

    # 创建弹性常数计算器
    calculator = ZeroKElasticConstantsCalculator(
        cell=supercell,
        potential=eam_potential,
        delta=0.001,  # 0.1% 应变
        num_steps=10,  # 7个应变点
        optimizer_type="LBFGS",
        optimizer_params={
            "ftol": 1e-12,  # 严格的能量收敛
            "gtol": 1e-10,  # 较严格的力收敛
            "maxls": 1000,  # 较大的线搜索步数
            "maxiter": 1000000,  # 足够大的迭代次数
        },
        save_path=output_dir,
    )
    logger.info("ZeroKElasticConstantsCalculator initialized.")

    # 计算弹性常数
    C = calculator.calculate_elastic_constants()
    logger.info("Elastic constants calculation completed.")

    # 检查立方晶体的基本对称性
    try:
        np.testing.assert_almost_equal(C[0, 0], C[1, 1], decimal=1)  # C11 = C22
        np.testing.assert_almost_equal(C[0, 0], C[2, 2], decimal=1)  # C11 = C33
        np.testing.assert_almost_equal(C[3, 3], C[4, 4], decimal=1)  # C44 = C55
        np.testing.assert_almost_equal(C[3, 3], C[5, 5], decimal=1)  # C44 = C66
        logger.info("Elastic constants matrix passed symmetry checks.")
    except AssertionError as e:
        logger.error("Elastic constants matrix symmetry checks failed.")
        raise e

    # 输出弹性常数
    logger.info("\nElastic constants (GPa):")
    logger.info(f"C11 = {C[0,0]:.3f} GPa")
    logger.info(f"C12 = {C[0,1]:.3f} GPa")
    logger.info(f"C44 = {C[3,3]:.3f} GPa")

    # 计算各向异性因子 A = 2C44/(C11-C12)
    A = 2 * C[3, 3] / (C[0, 0] - C[0, 1])
    logger.info(f"Anisotropy factor: A = {A:.3f}")

    # 计算体积模量 B = (C11 + 2C12)/3
    B = (C[0, 0] + 2 * C[0, 1]) / 3
    logger.info(f"Bulk modulus: B = {B:.3f} GPa")

    # 力学稳定性检查
    stability_checks = {
        "C11 > 0": C[0, 0] > 0,
        "C44 > 0": C[3, 3] > 0,
        "C11 > |C12|": C[0, 0] > abs(C[0, 1]),
        "C11 + 2C12 > 0": (C[0, 0] + 2 * C[0, 1]) > 0,
    }

    for check, result in stability_checks.items():
        logger.info(f"Stability Check - {check}: {result}")

    # 保存弹性常数矩阵和相关参数到文件
    results_file = os.path.join(output_dir, "elastic_constants_fcc_GPa.txt")
    with open(results_file, "w") as f:
        f.write("Elastic Constants Matrix (GPa):\n")
        np.savetxt(f, C, fmt="%.6e")
        f.write(f"\nAnisotropy factor (A = 2C44/(C11-C12)): {A:.6f}\n")
        f.write(f"Bulk modulus (B = (C11 + 2C12)/3): {B:.6f} GPa\n")
        for check, result in stability_checks.items():
            f.write(f"{check}: {result}\n")
        f.write("\nNote: R^2 values close to 1 indicate good fit quality.\n")
    logger.info(f"Elastic constants and related parameters saved to {results_file}")

    # 返回弹性常数矩阵
    return C


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting test for FCC Aluminum Elastic Constants Calculation.")
        C_fcc = test_elastic_constants_fcc_aluminum_eam()
        logger.info("Test completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}")
