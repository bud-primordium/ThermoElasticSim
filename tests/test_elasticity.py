import pytest
import numpy as np
from datetime import datetime
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.elasticity import ElasticConstantsCalculator
import logging


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
    log_filename = (
        f"./logs/test_elasticity_{current_time}.log"  # 生成带时间戳的日志文件名
    )

    # 创建文件处理器
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件日志级别
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    yield

    # 测试结束后移除处理器
    logger.removeHandler(ch)
    logger.removeHandler(fh)


def test_elastic_constants_calculator():
    """
    @brief 测试 ElasticConstantsCalculator 计算弹性常数
    """
    logger = logging.getLogger(__name__)
    logger.debug("Starting Elastic Constants Calculator Test.")

    # 创建简单的 BCC 晶格，4 个原子
    atoms = []
    lattice_constant = 10.0  # BCC 晶格参数，单位为 Å
    positions = [
        [0.0, 0.0, 0.0],  # 体心立方结构原子位置
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ]

    for pos in positions:
        atoms.append(
            Atom(
                id=len(atoms),
                symbol="Al",
                mass_amu=26.9815,
                position=np.array(pos) * lattice_constant,
            )
        )

    lattice_vectors = np.eye(3) * lattice_constant  # 简单的立方晶胞
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms, pbc_enabled=True)
    logger.debug(f"Created cell with {len(atoms)} atoms.")

    # 定义 Lennard-Jones 势能
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma  # Å
    lj_potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)
    logger.debug("Initialized Lennard-Jones potential.")

    # 创建 ElasticConstantsCalculator 实例
    elastic_calculator = ElasticConstantsCalculator(
        cell=cell,
        potential=lj_potential,
        delta=1e-2,
        optimizer_type="BFGS",  # 使用梯度下降优化器
    )
    logger.debug(
        "Initialized ElasticConstantsCalculator with Gradient Descent Optimizer."
    )

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

    logger.debug("Elastic Constants Calculator Test Passed.")
