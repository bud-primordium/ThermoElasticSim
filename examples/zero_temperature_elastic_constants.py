# examples/zero_temperature_elastic_constants.py

"""
@file zero_temperature_elastic_constants.py
@brief 零温下弹性常数计算示例。
"""

import numpy as np
from src.python.structure import CrystalStructure, Particle
from src.python.potentials import LennardJonesPotential
from src.python.optimizers import ConjugateGradientOptimizer
from src.python.deformation import Deformer
from src.python.stress_evaluator import LennardJonesStressEvaluator
from src.python.strain import StrainCalculator
from src.python.solver import ElasticConstantSolver
from src.python.utilities import TensorConverter


def initialize_crystal() -> CrystalStructure:
    """
    @brief 初始化晶体结构。

    @return CrystalStructure 实例。
    """
    particles = [
        Particle(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0]),
        Particle(id=2, symbol="Al", mass=26.9815, position=[1.8075, 1.8075, 1.8075]),
        # 添加更多粒子
    ]
    lattice_vectors = [[3.615, 0.0, 0.0], [0.0, 3.615, 0.0], [0.0, 0.0, 3.615]]
    crystal = CrystalStructure(lattice_vectors=lattice_vectors, particles=particles)
    return crystal


def main() -> None:
    """
    @brief 主函数，执行零温下弹性常数计算流程。
    """
    # 初始化晶体结构
    crystal = initialize_crystal()

    # 定义相互作用势能
    potential_params = {"epsilon": 0.0103, "sigma": 3.405}  # 示例参数  # 示例参数
    potential = LennardJonesPotential(parameters=potential_params, cutoff=5.0)

    # 结构优化
    optimizer = ConjugateGradientOptimizer()
    optimized_crystal = optimizer.optimize(crystal, potential)

    # 生成应变矩阵
    deformer = Deformer()
    deformation_matrices = deformer.generate_deformations(delta=0.01)

    # 选择应力计算器
    stress_evaluator = LennardJonesStressEvaluator()

    # 计算应变和应力数据
    strain_calculator = StrainCalculator()
    elastic_solver = ElasticConstantSolver()

    strain_data: list = []
    stress_data: list = []

    for F in deformation_matrices:
        # 应用应变
        deformed_crystal = deformer.apply_deformation(optimized_crystal, F)

        # 计算应力
        stress_voigt = stress_evaluator.compute_stress(deformed_crystal, potential)

        # 计算应变
        strain_voigt = strain_calculator.calculate_strain(F)

        strain_data.append(strain_voigt)
        stress_data.append(stress_voigt)

    # 求解弹性刚度矩阵
    C = elastic_solver.solve(strain_data, stress_data)

    # 打印结果
    print("零温下弹性刚度矩阵 C (Voigt形式):")
    print(C)


if __name__ == "__main__":
    main()
