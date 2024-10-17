# examples/calculate_elastic_constants.py

import numpy as np
from python.deformation import Deformer
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.optimizers import GradientDescentOptimizer
from python.mechanics import (
    StressCalculatorLJ,
    StrainCalculator,
    ElasticConstantsSolver,
)
from python.utils import TensorConverter
from python.visualization import Visualizer

# 定义参数
delta = 0.01  # 应变量
lattice_constant = 4.05  # Å
mass = 26.9815  # 原子量，单位 amu
mass *= 1.66054e-27  # 转换为 kg
epsilon = 0.0103  # eV
sigma = 2.55  # Å
cutoff = 2.5 * sigma

# 创建原子和晶胞
lattice_vectors = np.eye(3) * lattice_constant
position = np.array([0.0, 0.0, 0.0])
atom = Atom(id=0, symbol="Al", mass=mass, position=position)
cell = Cell(lattice_vectors, [atom])

# 初始化变形器
deformer = Deformer(delta)
deformation_matrices = deformer.generate_deformation_matrices()

# 初始化势能和优化器
potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)
optimizer = GradientDescentOptimizer(max_steps=1000, tol=1e-6, step_size=1e-4)

# 初始化应力和应变计算器
stress_calculator = StressCalculatorLJ()
strain_calculator = StrainCalculator()

strains = []
stresses = []

for i, F in enumerate(deformation_matrices):
    # 复制原始晶胞
    deformed_cell = cell.copy()
    # 施加变形
    deformer.apply_deformation(deformed_cell, F)
    # 优化结构
    optimizer.optimize(deformed_cell, potential)
    # 计算应力
    stress_tensor = stress_calculator.compute_stress(deformed_cell, potential)
    # 计算应变
    strain_tensor = strain_calculator.compute_strain(F)
    # 转换为 Voigt 表示
    stress_voigt = TensorConverter.to_voigt(stress_tensor)
    strain_voigt = TensorConverter.to_voigt(strain_tensor)
    strains.append(strain_voigt)
    stresses.append(stress_voigt)
    print(f"Deformation {i} completed.")

# 求解弹性常数
solver = ElasticConstantsSolver()
C = solver.solve(strains, stresses)
print("Elastic Constants Matrix C:")
print(C)

# 可视化应力-应变关系
visualizer = Visualizer()
visualizer.plot_stress_strain(np.array(strains), np.array(stresses))
