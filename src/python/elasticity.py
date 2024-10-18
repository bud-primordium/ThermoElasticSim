# src/python/elasticity.py

import numpy as np
import logging
from .mechanics import StressCalculatorLJ, StrainCalculator
from .deformation import Deformer
from .optimizers import GradientDescentOptimizer, BFGSOptimizer
from .utils import TensorConverter

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ElasticConstantsSolver:
    """
    @class ElasticConstantsSolver
    @brief 计算弹性常数的求解器类。
    """

    def solve(self, strains, stresses):
        """
        @brief 通过最小二乘法求解弹性常数矩阵。

        @param strains 应变数据列表，形状为 (N, 6)
        @param stresses 应力数据列表，形状为 (N, 6)
        @return 弹性常数矩阵，形状为 (6, 6)
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
        return C


class ElasticConstantsCalculator:
    """
    @class ElasticConstantsCalculator
    @brief 用于计算弹性常数的类。
    """

    def __init__(self, cell, potential, delta=1e-3, optimizer_type="BFGS"):
        """
        @param cell 晶胞对象
        @param potential 势能对象
        @param delta 变形大小
        @param optimizer_type 优化器类型，支持 'GD'（梯度下降）和 'BFGS'
        """
        self.cell = cell
        self.potential = potential
        self.delta = delta
        self.deformer = Deformer(delta)
        self.stress_calculator = StressCalculatorLJ()
        self.strain_calculator = StrainCalculator()
        if optimizer_type == "GD":
            self.optimizer = GradientDescentOptimizer(
                max_steps=1000, tol=1e-6, step_size=1e-3
            )
        elif optimizer_type == "BFGS":
            self.optimizer = BFGSOptimizer(tol=1e-6)
        else:
            raise ValueError("Unsupported optimizer type. Choose 'GD' or 'BFGS'.")

    def calculate_elastic_constants(self):
        """
        @brief 计算弹性常数矩阵。

        @return 弹性常数矩阵，形状为 (6, 6)。
        """
        logger.debug("Starting elastic constants calculation.")
        F_list = self.deformer.generate_deformation_matrices()
        strains = []
        stresses = []

        for idx, F in enumerate(F_list):
            logger.debug(f"Applying deformation {idx+1}/{len(F_list)}:")
            logger.debug(F)

            # Copy the initial cell
            deformed_cell = self.cell.copy()

            # Apply deformation
            deformed_cell.apply_deformation(F)
            logger.debug("Applied deformation to cell.")

            # Compute energy before optimization
            energy_before = self.potential.calculate_energy(deformed_cell)
            logger.debug(f"Energy before optimization: {energy_before} eV")

            # Optimize the structure
            self.optimizer.optimize(deformed_cell, self.potential)
            logger.debug("Optimization completed.")

            # Compute energy after optimization
            energy_after = self.potential.calculate_energy(deformed_cell)
            logger.debug(f"Energy after optimization: {energy_after} eV")

            # Check if optimization actually reduced the energy
            if energy_after >= energy_before:
                logger.warning(
                    f"Energy after optimization ({energy_after}) >= before ({energy_before})"
                )

            # Calculate forces (redundant if optimized)
            self.potential.calculate_forces(deformed_cell)
            logger.debug("Calculated forces after optimization.")

            # Compute stress tensor
            stress_tensor = self.stress_calculator.compute_stress(
                deformed_cell, self.potential
            )
            logger.debug(f"Computed stress tensor:")
            logger.debug(stress_tensor)

            # Compute strain tensor in Voigt notation
            strain_voigt = self.strain_calculator.compute_strain(F)
            logger.debug(f"Computed strain (Voigt): {strain_voigt}")

            # Convert stress tensor to Voigt notation
            stress_voigt = TensorConverter.to_voigt(stress_tensor)
            logger.debug(f"Converted stress to Voigt notation: {stress_voigt}")

            strains.append(strain_voigt)
            stresses.append(stress_voigt)

        # Solve for elastic constants using the solver
        logger.debug("Solving for elastic constants.")
        elastic_solver = ElasticConstantsSolver()
        C = elastic_solver.solve(strains, stresses)
        logger.debug(f"Elastic constants matrix (eV/Å^3 / strain):\n{C}")

        # Convert to GPa
        C_in_GPa = C * 160.21766208  # eV/Å^3 转 GPa
        logger.debug(f"Elastic constants matrix (GPa):\n{C_in_GPa}")

        return C_in_GPa
