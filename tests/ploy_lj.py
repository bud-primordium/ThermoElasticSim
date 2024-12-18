import numpy as np
import matplotlib.pyplot as plt
from python.potentials import LennardJonesPotential
from python.structure import Cell, Atom

# 创建 Lennard-Jones 势
epsilon = 0.0103  # eV
sigma = 2.55  # Å
cutoff = 5.1  # Å
lj_potential = LennardJonesPotential(epsilon, sigma, cutoff)

# 创建模拟单元格
atoms = [
    Atom(id=0, symbol="Al", mass_amu=26.9815, position=np.array([0.0, 0.0, 0.0])),
    Atom(id=1, symbol="Al", mass_amu=26.9815, position=np.array([2.5, 0.0, 0.0])),
]
cell = Cell(atoms=atoms, lattice_vectors=np.eye(3) * 30.0, pbc_enabled=True)

# 准备测试距离
distances = np.linspace(2.4, 6.0, 1000)
forces = []
energies = []

# 修改原子位置，计算不同距离下的力和能量
for d in distances:
    cell.atoms[1].position = np.array([d, 0.0, 0.0])

    # 计算力和能量
    lj_potential.calculate_forces(cell)
    forces.append(cell.atoms[1].force[0])  # 只记录x方向力
    energies.append(lj_potential.calculate_energy(cell))

# Lennard-Jones 力和能量公式
theoretical_forces = [
    24.0 * epsilon * ((2 * (sigma / d) ** 12) - (sigma / d) ** 6) / d for d in distances
]
theoretical_energies = [
    4.0 * epsilon * ((sigma / d) ** 12 - (sigma / d) ** 6) for d in distances
]
# 绘制力-距离和能量-距离曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(distances, forces, label="Computed Force")
plt.plot(distances, theoretical_forces, label="Theoretical Force", linestyle="--")
plt.title("Force vs. Distance")
plt.xlabel("Distance (Å)")
plt.ylabel("Force (eV/Å)")

plt.subplot(1, 2, 2)
plt.plot(distances, energies, label="Computed Energy")
plt.plot(distances, theoretical_energies, label="Theoretical Energy", linestyle="--")
plt.title("Energy vs. Distance")
plt.xlabel("Distance (Å)")
plt.ylabel("Energy (eV)")

plt.tight_layout()
plt.show()
