import numpy as np
from python.structure import Atom, Cell
from python.potentials import EAMAl1Potential
from python.mechanics import StressCalculatorEAM
from python.optimizers import BFGSOptimizer, LBFGSOptimizer, GradientDescentOptimizer
import logging


def test_simple_strain_stress_response():
    """
    使用两阶段优化的应变-应力响应测试
    """
    logger = logging.getLogger(__name__)

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

    # 创建基本晶胞并构建超胞
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    supercell = cell.build_supercell((2, 2, 2))

    logger.info(f"Created supercell with {supercell.num_atoms} atoms")
    logger.info(f"Cell dimensions: {np.diag(supercell.lattice_vectors)} Å")

    # 创建两个优化器，使用不同的参数
    # 粗优化：使用宽松的标准
    coarse_optimizer = LBFGSOptimizer(
        ftol=1e-5,  # 比默认值宽松
        gtol=1e-4,  # 比默认值宽松
        maxiter=1000,  # 较少的迭代次数
        maxls=50,  # 较多的线搜索步数
    )

    # 精优化：使用严格的标准
    fine_optimizer = LBFGSOptimizer(
        ftol=1e-8,  # 比默认值严格
        gtol=1e-5,  # 比默认值严格
        maxls=100,  # 较多的线搜索步数
        maxiter=10000,  # 允许更多迭代
    )

    potential = EAMAl1Potential()
    stress_calculator = StressCalculatorEAM()

    # 优化初始结构（两阶段优化）
    logger.info("Optimizing initial structure (coarse stage)...")
    converged, _ = coarse_optimizer.optimize(supercell, potential)
    if not converged:
        logger.warning("Initial coarse optimization did not converge!")

    logger.info("Optimizing initial structure (fine stage)...")
    converged, _ = fine_optimizer.optimize(supercell, potential)
    if not converged:
        logger.warning("Initial fine optimization did not converge!")

    # 计算初始应力
    initial_stress = stress_calculator.compute_stress(supercell, potential)
    logger.info(f"Initial stress tensor (eV/Å³):\n{initial_stress}")
    logger.info(
        f"Initial potential energy (eV): {potential.calculate_energy(supercell)}"
    )

    # 测试不同方向的应变
    strains = [0.0001]  # 0.01%的应变

    # 定义要测试的变形类型
    deformation_types = [
        ("xx", lambda e: np.array([[1 + e, 0, 0], [0, 1, 0], [0, 0, 1]])),
        ("yy", lambda e: np.array([[1, 0, 0], [0, 1 + e, 0], [0, 0, 1]])),
        ("zz", lambda e: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1 + e]])),
        ("xy", lambda e: np.array([[1, e, 0], [e, 1, 0], [0, 0, 1]])),
        ("yz", lambda e: np.array([[1, 0, 0], [0, 1, e], [0, e, 1]])),
        ("xz", lambda e: np.array([[1, 0, e], [0, 1, 0], [e, 0, 1]])),
    ]

    # 测试每种变形
    for deform_type, F_generator in deformation_types:
        logger.info(f"\nTesting {deform_type} deformation:")
        for strain in strains:
            # 构造变形矩阵
            F = F_generator(strain)

            # 创建变形后的晶胞
            deformed_cell = supercell.copy()
            deformed_cell.apply_deformation(F)

            # 两阶段优化变形后的结构
            logger.info(
                f"Optimizing structure after {deform_type} strain {strain:.6f} (coarse stage)..."
            )
            converged, _ = coarse_optimizer.optimize(deformed_cell, potential)
            if not converged:
                logger.warning(
                    f"Coarse optimization after {deform_type} strain {strain:.6f} did not converge!"
                )

            logger.info(
                f"Optimizing structure after {deform_type} strain {strain:.6f} (fine stage)..."
            )
            converged, _ = fine_optimizer.optimize(deformed_cell, potential)
            if not converged:
                logger.warning(
                    f"Fine optimization after {deform_type} strain {strain:.6f} did not converge!"
                )

            # 计算应力
            stress = stress_calculator.compute_stress(deformed_cell, potential)

            # 计算应力差（相对于初始应力）
            stress_difference = stress - initial_stress

            logger.info(f"Strain {strain:.6f}:")
            logger.info(f"Deformation matrix F:\n{F}")
            logger.info(f"Stress tensor (eV/Å³):\n{stress}")
            logger.info(f"Stress difference (eV/Å³):\n{stress_difference}")
            logger.info(
                f"Potential energy (eV): {potential.calculate_energy(deformed_cell)}"
            )

            # 检查应力张量的对称性
            np.testing.assert_almost_equal(
                stress_difference,
                stress_difference.T,
                decimal=5,
                err_msg=f"Stress difference tensor not symmetric for {deform_type} strain",
            )

            # 记录主应力或剪切应力响应
            if deform_type in ["xx", "yy", "zz"]:
                idx = {"xx": 0, "yy": 1, "zz": 2}[deform_type]
                main_stress = stress_difference[idx, idx]
                logger.info(
                    f"{deform_type} strain {strain:.6f} -> main stress difference {main_stress:.6e}"
                )
            elif deform_type in ["xy", "yz", "xz"]:
                idx_map = {"xy": (0, 1), "yz": (1, 2), "xz": (0, 2)}
                i, j = idx_map[deform_type]
                shear_stress = stress_difference[i, j]
                logger.info(
                    f"{deform_type} strain {strain:.6f} -> shear stress difference {shear_stress:.6e}"
                )
