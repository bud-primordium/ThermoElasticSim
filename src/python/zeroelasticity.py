# 文件名: zeroelasticity.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 实现用于计算弹性常数的求解器和计算类（零温条件下）。

"""
弹性常数模块（零温）

包含 ZeroKElasticConstantsSolver 和 ZeroKElasticConstantsCalculator 类，
用于通过应力应变数据计算材料的弹性常数
"""

import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from .mechanics import StressCalculatorLJ, StrainCalculator
from .deformation import Deformer
from .optimizers import GradientDescentOptimizer, BFGSOptimizer
from .utils import TensorConverter, EV_TO_GPA  # 导入单位转换因子

# 配置日志记录
logger = logging.getLogger(__name__)


class ZeroKElasticConstantsSolver:
    """
    计算弹性常数的求解器类（零温）
    """

    def solve(self, strains, stresses):
        """
        通过最小二乘法求解弹性常数矩阵

        Parameters
        ----------
        strains : array_like
            应变数据，形状为 (N, 6)
        stresses : array_like
            应力数据，形状为 (N, 6)

        Returns
        -------
        numpy.ndarray
            弹性常数矩阵，形状为 (6, 6)
        """
        strains = np.array(strains)
        stresses = np.array(stresses)

        # 检查输入数据维度
        if strains.ndim != 2 or stresses.ndim != 2:
            raise ValueError("Strains and stresses must be 2D arrays.")
        if strains.shape[0] != stresses.shape[0]:
            raise ValueError("Number of strain and stress samples must be equal.")
        if strains.shape[1] != 6 or stresses.shape[1] != 6:
            raise ValueError("Strains and stresses must have 6 components each.")

        logger.debug("Solving elastic constants using least squares.")
        C, residuals, rank, s = np.linalg.lstsq(strains, stresses, rcond=None)
        logger.debug(f"Elastic constants matrix (before conversion):\n{C}")
        return C.T  # 返回转置后的矩阵


class ZeroKElasticConstantsCalculator:
    """
    计算弹性常数的类（零温）

    Parameters
    ----------
    cell : Cell
        晶胞对象
    potential : Potential
        势能对象
    delta : float, optional
        变形大小，默认为 1e-3
    optimizer_type : str, optional
        优化器类型，支持 'GD'（梯度下降）和 'BFGS'，默认为 'GD'
    """

    def __init__(self, cell, potential, delta=1e-3, optimizer_type="GD"):
        self.cell = cell
        self.potential = potential
        self.delta = delta  # 指的是形变
        self.deformer = Deformer(delta)
        self.stress_calculator = StressCalculatorLJ()
        self.strain_calculator = StrainCalculator()

        if optimizer_type == "GD":
            self.optimizer = GradientDescentOptimizer(
                max_steps=10000, tol=1e-5, step_size=1e-3, energy_tol=1e-5
            )
        elif optimizer_type == "BFGS":
            self.optimizer = BFGSOptimizer(tol=1e-5, maxiter=10000)
        else:
            raise ValueError("Unsupported optimizer type. Choose 'GD' or 'BFGS'.")

    def calculate_initial_stress(self):
        """
        计算初始结构的应力，在优化之前验证应力计算是否正确

        Returns
        -------
        numpy.ndarray
            初始应力张量
        """
        logger.debug("Calculating initial stress before optimization.")
        initial_stress = self.stress_calculator.compute_stress(
            self.cell, self.potential
        )
        logger.debug(f"Initial stress tensor before optimization:\n{initial_stress}")
        return initial_stress

    def optimize_initial_structure(self):
        """在施加变形前对结构进行一次优化，使得初始结构的应力为零"""
        logger.debug("Starting initial structure optimization.")
        self.optimizer.optimize(self.cell, self.potential)
        logger.debug("Initial structure optimization completed.")

    def calculate_stress_strain(self, F, deformation_index):
        """
        对指定的变形矩阵计算应力和应变

        Parameters
        ----------
        F : numpy.ndarray
            变形矩阵
        deformation_index : int
            变形矩阵的编号，便于日志追踪

        Returns
        -------
        tuple
            包含应变（Voigt 表示）和应力（Voigt 表示）的元组
        """
        logger.info(f"Processing deformation matrix #{deformation_index}")
        logger.debug(f"Deformation matrix F:\n{F}")

        # 复制初始晶胞
        deformed_cell = self.cell.copy()

        # 施加变形
        deformed_cell.apply_deformation(F)
        logger.debug(f"Applied deformation to cell #{deformation_index}.")

        # 为每个任务创建独立的优化器实例
        if isinstance(self.optimizer, GradientDescentOptimizer):
            optimizer = GradientDescentOptimizer(
                max_steps=self.optimizer.max_steps,
                tol=self.optimizer.tol,
                step_size=self.optimizer.step_size,
                energy_tol=self.optimizer.energy_tol,
                beta=self.optimizer.beta,
            )
        elif isinstance(self.optimizer, BFGSOptimizer):
            optimizer = BFGSOptimizer(
                tol=self.optimizer.tol,
                maxiter=self.optimizer.maxiter,
            )

        # 对变形后的结构进行优化
        optimizer.optimize(deformed_cell, self.potential)
        logger.debug(f"Optimized deformed structure #{deformation_index}.")

        # 计算应力张量
        stress_tensor = self.stress_calculator.compute_stress(
            deformed_cell, self.potential
        )
        logger.info(
            f"Computed stress tensor for deformation #{deformation_index}:\n{stress_tensor}"
        )

        # 计算应变张量
        strain_voigt = self.strain_calculator.compute_strain(F)
        logger.info(
            f"Computed strain (Voigt) for deformation #{deformation_index}: {strain_voigt}"
        )

        # 转换应力为 Voigt 表示法
        stress_voigt = TensorConverter.to_voigt(stress_tensor)
        logger.info(
            f"Converted stress to Voigt notation for deformation #{deformation_index}: {stress_voigt}"
        )

        # 输出位置以确认变形后的结构
        final_positions = deformed_cell.get_positions()
        logger.debug(
            f"Final atom positions for deformation #{deformation_index}:\n{final_positions}"
        )

        return strain_voigt, stress_voigt

    def calculate_elastic_constants(self):
        """
        计算弹性常数矩阵

        Returns
        -------
        numpy.ndarray
            弹性常数矩阵，形状为 (6, 6)，单位为 GPa
        """
        logger.info("Starting elastic constants calculation.")

        # 在优化前计算初始应力
        self.calculate_initial_stress()

        # 优化初始结构
        self.optimize_initial_structure()

        # 生成六个变形矩阵
        F_list = self.deformer.generate_deformation_matrices()
        strains = []
        stresses = []

        # 并行计算每个应变的应力和应变
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda i_F: self.calculate_stress_strain(i_F[1], i_F[0]),
                enumerate(F_list),
            )

        for strain, stress in results:
            strains.append(strain)
            stresses.append(stress)

        # 求解弹性常数矩阵
        logger.debug("Solving for elastic constants.")
        elastic_solver = ZeroKElasticConstantsSolver()
        C = elastic_solver.solve(strains, stresses)
        logger.info(f"Elastic constants matrix (eV/Å^3 / strain):\n{C}")

        # 单位转换为 GPa
        C_in_GPa = C * EV_TO_GPA
        logger.info(f"Elastic constants matrix (GPa):\n{C_in_GPa}")

        return C_in_GPa
