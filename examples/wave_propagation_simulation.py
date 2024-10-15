# examples/wave_propagation_simulation.py

"""
@file wave_propagation_simulation.py
@brief 弹性波传播模拟示例。
"""

import numpy as np
from src.python.structure import Cell, Atom
from src.python.potentials import LennardJonesPotential
from src.python.optimizers import ConjugateGradientOptimizer
from src.python.deformation import Deformer
from src.python.stress_evaluator import LennardJonesStressEvaluator
from src.python.strain import StrainCalculator
from src.python.solver import ElasticConstantSolver
from src.python.md_simulation import MDSimulator
from src.python.visualization import Visualizer
from src.python.utilities import TensorConverter


def initialize_cell() -> Cell:
    """
    @brief 初始化晶体结构。

    @return Cell 实例。
    """
    atoms = [
        Atom(id=1, symbol="Al", mass=26.9815, position=[0.0, 0.0, 0.0]),
        Atom(id=2, symbol="Al", mass=26.9815, position=[1.8075, 1.8075, 1.8075]),
        # 添加更多原子
    ]
    lattice_vectors = [[3.615, 0.0, 0.0], [0.0, 3.615, 0.0], [0.0, 0.0, 3.615]]
    cell = Cell(lattice_vectors=lattice_vectors, atoms=atoms)
    return cell


def main() -> None:
    """
    @brief 主函数，执行弹性波传播模拟流程。
    """
    # 初始化晶体结构
    cell = initialize_cell()

    # 定义相互作用势能
    potential_params = {"epsilon": 0.0103, "sigma": 3.405}  # 示例参数  # 示例参数
    potential = LennardJonesPotential(parameters=potential_params, cutoff=5.0)

    # 结构优化
    optimizer = ConjugateGradientOptimizer()
    optimized_cell = optimizer.optimize(cell, potential)

    # 初始化分子动力学模拟器
    md_simulator = MDSimulator(
        cell_structure=optimized_cell,
        potential=potential,
        temperature=300.0,  # K
        pressure=0.0,  # GPa
        timestep=1.0e-3,  # ps
        thermostat="Nosé-Hoover",
        barostat="NoBarostat",
    )

    # 运行分子动力学模拟
    md_simulator.run_simulation(steps=10000)

    # 获取模拟后的晶体结构
    # 假设 MDSimulator 更新了 cell_structure 属性
    simulated_cell = md_simulator.cell_structure

    # 可视化晶体结构
    visualizer = Visualizer()
    visualizer.plot_cell_structure(simulated_cell)

    # 模拟弹性波传播
    # 这里简化为对晶体结构施加微小扰动，然后观察其响应
    deformer = Deformer()
    F = np.identity(3) + 1e-3 * np.random.randn(3, 3)  # 微小随机应变
    deformed_cell = deformer.apply_deformation(simulated_cell, F)

    # 计算应力
    stress_evaluator = LennardJonesStressEvaluator()
    stress_voigt = stress_evaluator.compute_stress(deformed_cell, potential)

    # 计算应变
    strain_calculator = StrainCalculator()
    strain_voigt = strain_calculator.calculate_strain(F)

    # 打印应力和应变
    print("施加的应变 (Voigt形式):")
    print(strain_voigt)
    print("计算的应力 (Voigt形式):")
    print(stress_voigt)

    # 可视化应力-应变关系
    strain_data = [strain_voigt]
    stress_data = [stress_voigt]
    visualizer.plot_stress_strain(np.array(strain_data), np.array(stress_data))


if __name__ == "__main__":
    main()
