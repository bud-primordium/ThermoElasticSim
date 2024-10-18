from python.structure import Atom, Cell
from python.potentials import LennardJonesPotential
import numpy as np


def manual_test_lj():
    epsilon = 0.0103  # eV
    sigma = 2.55  # Å
    cutoff = 2.5 * sigma  # Å

    lj_potential = LennardJonesPotential(epsilon, sigma, cutoff)

    # 创建两个原子
    atom1 = Atom(id=0, symbol="Al", mass_amu=26.9815, position=[0.0, 0.0, 0.0])
    atom2 = Atom(
        id=1, symbol="Al", mass_amu=26.9815, position=[3.0, 0.0, 0.0]
    )  # 初始距离为3.0 Å

    cell = Cell(
        lattice_vectors=np.eye(3) * 10.0, atoms=[atom1, atom2], pbc_enabled=False
    )

    # 计算能量和力
    energy = lj_potential.calculate_energy(cell)
    lj_potential.calculate_forces(cell)
    force1 = atom1.force
    force2 = atom2.force

    print(f"Energy: {energy} eV")
    print(f"Force on Atom 0: {force1} eV/A")
    print(f"Force on Atom 1: {force2} eV/A")


if __name__ == "__main__":
    manual_test_lj()
