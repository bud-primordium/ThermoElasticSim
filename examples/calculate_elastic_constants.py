# examples/calculate_elastic_constants.py

import numpy as np
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.optimizers import QuickminOptimizer
from python.deformation import Deformer
from python.md import MDSimulator, VelocityVerletIntegrator, NoseHooverThermostat
from python.mechanics import (
    StressCalculatorLJ,
    StrainCalculator,
    ElasticConstantsSolver,
)
from python.utils import TensorConverter

# 定义 Al 的晶格常数和晶格矢量
lattice_constant = 4.05e-10  # 米
lattice_vectors = 0.5 * lattice_constant * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

# 定义原子位置（以晶格矢量为基）
fractional_positions = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]

atoms = []
for i, frac_pos in enumerate(fractional_positions):
    position = np.dot(frac_pos, lattice_vectors)
    mass_kg = 26.9815 / (6.02214076e23) * 1e-3  # 质量单位为 kg
    atom = Atom(id=i, symbol="Al", mass=mass_kg, position=position)
    atoms.append(atom)

cell = Cell(lattice_vectors, atoms)

# 定义 Al 的 Lennard-Jones 势参数
sigma = 2.55e-10  # 米
epsilon = 6.774e-21  # 焦耳
cutoff = 2.5 * sigma  # 截断半径

potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

# 结构优化
optimizer = QuickminOptimizer(max_steps=500, tol=1e-8)
optimizer.optimize(cell, potential)

# 生成变形矩阵
delta = 0.01  # 变形幅度
deformer = Deformer(delta)
deformation_matrices = deformer.generate_deformation_matrices()

# 模拟参数
temperatures = [300]  # 温度列表（单位：开尔文）

for T in temperatures:
    print(f"计算温度 {T} K 下的弹性常数")
    strains = []
    stresses = []
    for F in deformation_matrices:
        # 创建变形后的晶胞
        deformed_cell = cell.copy()
        deformer.apply_deformation(deformed_cell, F)

        # 分子动力学模拟
        integrator = VelocityVerletIntegrator()
        thermostat = NoseHooverThermostat(target_temperature=T, time_constant=100)
        md_simulator = MDSimulator(deformed_cell, potential, integrator, thermostat)
        md_simulator.run(steps=1000, dt=1e-15)

        # 计算应力和应变
        stress_calculator = StressCalculatorLJ()
        stress_tensor = stress_calculator.compute_stress(deformed_cell, potential)
        strain_calculator = StrainCalculator()
        strain_tensor = strain_calculator.compute_strain(F)

        # 转换为 Voigt 表示
        stress_voigt = TensorConverter.to_voigt(stress_tensor)
        strain_voigt = TensorConverter.to_voigt(strain_tensor)

        # 打印应力和应变
        print("应变张量（Voigt）：", strain_voigt)
        print("应力张量（Voigt）：", stress_voigt)

        strains.append(strain_voigt)
        stresses.append(stress_voigt)

    # 求解弹性常数
    solver = ElasticConstantsSolver()
    C = solver.solve(strains, stresses)
    print(f"温度 {T} K 下的弹性常数矩阵：")
    print(C)
