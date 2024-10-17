# examples/test_cpp_calculate_forces.py

import numpy as np
import sys
import os

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

from python.potentials import LennardJonesPotential
from python.structure import Atom, Cell


def main():
    # 定义参数
    sigma = 2.55e-10  # m
    epsilon = 6.774e-21  # J
    cutoff = 2.5 * sigma
    r_eq = 2 ** (1 / 6) * sigma  # 平衡距离

    # 创建两个铝原子，距离为 r_eq
    lattice_vectors = np.eye(3) * 4.05e-10  # 米
    mass = 26.9815 / (6.02214076e23) * 1e-3  # kg
    position1 = np.array([0.0, 0.0, 0.0])
    position2 = np.array([r_eq, 0.0, 0.0])
    atom1 = Atom(id=0, symbol="Al", mass=mass, position=position1)
    atom2 = Atom(id=1, symbol="Al", mass=mass, position=position2)
    cell = Cell(
        lattice_vectors, [atom1, atom2], pbc_enabled=False
    )  # 禁用周期性边界条件

    # 定义 Lennard-Jones 势
    potential = LennardJonesPotential(epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    # 计算力
    potential.calculate_forces(cell)

    # 打印力
    for atom in cell.atoms:
        print(f"Atom ID {atom.id} Force: {atom.force}")


if __name__ == "__main__":
    main()
