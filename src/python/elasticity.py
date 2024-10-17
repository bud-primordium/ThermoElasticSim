# src/python/elasticity.py

import numpy as np
from .mechanics import StressCalculatorLJ, ElasticConstantsSolver
from .deformation import Deformer
from .optimizers import GradientDescentOptimizer
from .utils import TensorConverter


class ElasticConstantsCalculator:
    """
    @class ElasticConstantsCalculator
    @brief 用于计算弹性常数的类
    """

    def __init__(self, cell, potential, delta=1e-3):
        self.cell = cell
        self.potential = potential
        self.delta = delta
        self.deformer = Deformer(delta)
        self.stress_calculator = StressCalculatorLJ()
        self.optimizer = GradientDescentOptimizer(
            max_steps=1000, tol=1e-6, step_size=1e-4
        )

    def calculate_elastic_constants(self):
        """
        @brief 计算弹性常数矩阵

        @return 弹性常数矩阵，形状为 (6, 6)
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
            strain_tensor = 0.5 * (np.dot(F.T, F) - np.identity(3))
            # 转换为 Voigt 表示
            strain_voigt = TensorConverter.to_voigt(strain_tensor)
            stress_voigt = TensorConverter.to_voigt(stress_tensor)
            strains.append(strain_voigt)
            stresses.append(stress_voigt)

        # 拟合弹性常数矩阵
        strains = np.array(strains)
        stresses = np.array(stresses)
        elastic_solver = ElasticConstantsSolver()
        C = elastic_solver.solve(strains, stresses)
        return C
