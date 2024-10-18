import numpy as np
import matplotlib.pyplot as plt
from python.potentials import LennardJonesPotential
from python.structure import Cell, Atom  # 假设已有的结构体定义

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
box_lengths = np.array([30.0, 30.0, 30.0])  # 立方体盒子

cell = Cell(atoms=atoms, lattice_vectors=np.eye(3) * 20.0, pbc_enabled=True)

# 准备测试距离
distances = np.linspace(2.0, 6.0, 1000)
forces = []
energies = []

# 修改原子位置，计算不同距离下的力和能量
for d in distances:
    cell.atoms[1].position = np.array([d, 0.0, 0.0])

    # 计算力和能量
    lj_potential.calculate_forces(cell)
    forces.append(cell.atoms[0].force[0])  # 只记录x方向力
    energies.append(lj_potential.calculate_energy(cell))

# 绘制力-距离和能量-距离曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(distances, forces)
plt.title("Force vs. Distance")
plt.xlabel("Distance (Å)")
plt.ylabel("Force (eV/Å)")

plt.subplot(1, 2, 2)
plt.plot(distances, energies)
plt.title("Energy vs. Distance")
plt.xlabel("Distance (Å)")
plt.ylabel("Energy (eV)")

plt.tight_layout()
plt.show()
