# scripts/calculate_elastic_constants.py

import numpy as np
from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
from python.elasticity import ElasticConstantsCalculator


def main():
    # 创建初始晶胞
    lattice_constant = 4.05  # Å
    lattice_vectors = np.array(
        [[lattice_constant, 0, 0], [0, lattice_constant, 0], [0, 0, lattice_constant]]
    )
    mass_amu = 26.9815  # amu
    # 构建面心立方（FCC）结构的原子位置
    positions = [
        [0, 0, 0],
        [0.5 * lattice_constant, 0.5 * lattice_constant, 0],
        [0.5 * lattice_constant, 0, 0.5 * lattice_constant],
        [0, 0.5 * lattice_constant, 0.5 * lattice_constant],
    ]
    atoms = [
        Atom(id=i, symbol="Al", mass_amu=mass_amu, position=pos)
        for i, pos in enumerate(positions)
    ]
    cell = Cell(lattice_vectors, atoms)

    # 定义 Lennard-Jones 势
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma
    potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    # 创建弹性常数计算器
    calculator = ElasticConstantsCalculator(cell, potential)
    C = calculator.calculate_elastic_constants()
    print("弹性常数矩阵 (GPa):")
    print(C * 160.21766208)  # 将 eV/Å³ 转换为 GPa


if __name__ == "__main__":
    main()
