# src/python/elasticity.py

import numpy as np
from .mechanics import StressCalculatorLJ, StrainCalculator
from .deformation import Deformer
from .optimizers import GradientDescentOptimizer, BFGSOptimizer
from .utils import TensorConverter


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

        # 使用最小二乘法求解 C * strains.T = stresses.T
        # 这里 C 是一个 6x6 矩阵，每一行对应一个应力分量的线性组合
        C = np.linalg.lstsq(strains, stresses, rcond=None)[0]
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
                max_steps=1000, tol=1e-6, step_size=1e-4
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
        F_list = self.deformer.generate_deformation_matrices()
        strains = []
        stresses = []

        for F in F_list:
            # 复制初始晶胞
            deformed_cell = self.cell.copy()
            # 施加变形
            deformed_cell.apply_deformation(F)
            # 优化结构
            self.optimizer.optimize(deformed_cell, self.potential)
            # 计算应力
            self.potential.calculate_forces(deformed_cell)
            stress_tensor = self.stress_calculator.compute_stress(
                deformed_cell, self.potential
            )
            # 计算应变
            strain_voigt = self.strain_calculator.compute_strain(F)
            # 转换应力张量为 Voigt 表示
            stress_voigt = TensorConverter.to_voigt(stress_tensor)
            strains.append(strain_voigt)
            stresses.append(stress_voigt)

        # 使用弹性常数求解器
        elastic_solver = ElasticConstantsSolver()
        C = elastic_solver.solve(strains, stresses)
        return C
