# 文件名: test_zeroelasticity.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 测试用于计算零温弹性常数的求解器和计算类。

import pytest
import numpy as np
from datetime import datetime
from itertools import product
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.zeroelasticity import ZeroKElasticConstantsCalculator  # 更新导入路径和类名
from python.utils import NeighborList
import logging
import os


# 配置日志
@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """
    配置日志以在测试期间输出到控制台和文件。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # 控制台日志级别
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 日志文件路径
    log_directory = "./logs/elasticity/"
    log_filename = f"{log_directory}/0K_{current_time}.log"  # 生成带时间戳的日志文件名

    # 确保日志目录存在
    os.makedirs(log_directory, exist_ok=True)

    # 创建文件处理器
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件日志级别
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    yield

    # 测试结束后移除处理器
    logger.removeHandler(ch)
    logger.removeHandler(fh)


def generate_fcc_supercell(lattice_constant, repetitions):
    """
    生成面心立方 (FCC) 结构的超胞原子位置。

    Parameters
    ----------
    lattice_constant : float
        FCC 晶格常数，单位 Å。
    repetitions : int
        每个方向上的单位晶胞重复次数。

    Returns
    -------
    list of list of float
        原子位置列表。
    """
    # FCC 单位晶胞的标准原子位置（分数坐标）
    base_positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]

    positions = []
    for i, j, k in product(range(repetitions), repeat=3):
        for pos in base_positions:
            cartesian = (np.array([i, j, k]) + np.array(pos)) * lattice_constant
            positions.append(cartesian.tolist())

    return positions


def test_zeroK_elastic_constants():
    """
    @brief 测试零温度下的弹性常数计算，确保在变形后进行结构优化。
    """
    logger = logging.getLogger(__name__)
    logger.debug("Starting 0K Elastic Constants Calculator Test.")

    lattice_constant = 4.05  # Å
    repetitions = 1

    # 生成 1x1x1 超胞的原子位置
    positions = generate_fcc_supercell(lattice_constant, repetitions)
    atoms = []
    for idx, pos in enumerate(positions):
        atoms.append(
            Atom(
                id=idx,
                symbol="Al",
                mass_amu=26.9815,
                position=pos,
            )
        )

    # 更新晶格矢量
    lattice_vectors = np.eye(3) * lattice_constant * repetitions

    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    logger.debug(f"Created cell with {len(atoms)} atoms.")

    # 定义 Lennard-Jones 势能
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma  # Å
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)
    logger.debug("Initialized Lennard-Jones potential.")

    # 创建邻居列表并关联到势能函数
    neighbor_list = NeighborList(cutoff=lj_potential.cutoff)
    neighbor_list.build(cell)
    lj_potential.set_neighbor_list(neighbor_list)
    logger.debug("Neighbor list built and set for potential.")

    # 创建 ZeroKElasticConstantsCalculator 实例
    elastic_calculator = ZeroKElasticConstantsCalculator(
        cell=cell,
        potential=lj_potential,
        delta=1e-4,
        optimizer_type="BFGS",  # 使用BFGS优化器
    )
    logger.debug("Initialized ZeroKElasticConstantsCalculator with BFGS Optimizer.")

    # 计算弹性常数
    C_in_GPa = elastic_calculator.calculate_elastic_constants()
    logger.debug("Elastic constants calculation completed.")

    # 输出计算的弹性常数
    logger.debug("Computed Elastic Constants (GPa):")
    logger.debug(C_in_GPa)

    # 检查 C_in_GPa 是一个 6x6 的矩阵
    assert C_in_GPa.shape == (6, 6), "弹性常数矩阵形状不匹配。"
    logger.debug("Shape of Elastic Constants matrix is correct (6x6).")

    # 检查对称性
    symmetry = np.allclose(C_in_GPa, C_in_GPa.T, atol=1e-3)
    logger.debug(f"Symmetry check: {symmetry}")
    assert symmetry, "弹性常数矩阵不是对称的。"

    # 检查对角元素为正
    for i in range(6):
        logger.debug(f"C_in_GPa[{i},{i}] = {C_in_GPa[i, i]}")
        assert C_in_GPa[i, i] > 0, f"弹性常数 C[{i},{i}] 不是正值。"

    # 检查非对角元素在合理范围内
    for i in range(6):
        for j in range(i + 1, 6):
            logger.debug(f"C_in_GPa[{i},{j}] = {C_in_GPa[i, j]}")
            assert (
                0.0 <= C_in_GPa[i, j] <= 100.0
            ), f"弹性常数 C[{i},{j}] 不在合理范围内。"

    logger.debug("ZeroK Elastic Constants Calculator Test Passed.")
