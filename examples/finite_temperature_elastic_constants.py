# examples/finite_temperature_elastic_constants.py

"""
@file finite_temperature_elastic_constants.py
@brief 有限温度下弹性常数计算示例。
"""

from src.python.structure import Cell, Atom
from src.python.potentials import LennardJonesPotential
from src.python.optimizers import ConjugateGradientOptimizer
from src.python.deformation import Deformer
from src.python.stress_evaluator import LennardJonesStressEvaluator
from src.python.strain import StrainCalculator
from src.python.solver import ElasticConstantSolver
from src.python.md_simulation import MDSimulator
from python.utils import TensorConverter


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
    @brief 主函数，执行有限温度下弹性常数计算流程。
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
        deformed_cell = deformer.apply_deformation(simulated_cell, F)

        # 计算应力
        stress_voigt = stress_evaluator.compute_stress(deformed_cell, potential)

        # 计算应变
        strain_voigt = strain_calculator.calculate_strain(F)

        strain_data.append(strain_voigt)
        stress_data.append(stress_voigt)

    # 求解弹性刚度矩阵
    C = elastic_solver.solve(strain_data, stress_data)

    # 打印结果
    print("有限温度下弹性刚度矩阵 C (Voigt形式):")
    print(C)


if __name__ == "__main__":
    main()
