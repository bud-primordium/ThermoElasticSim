import numpy as np
import matplotlib.pyplot as plt
from python.potentials import EAMAl1Potential
from python.structure import Cell, Atom

# 创建 EAM Al1 势
eam_potential = EAMAl1Potential()

# 创建模拟单元格
atoms = [
    Atom(id=0, symbol="Al", mass_amu=26.9815, position=np.array([0.0, 0.0, 0.0])),
    Atom(id=1, symbol="Al", mass_amu=26.9815, position=np.array([2.8, 0.0, 0.0])),
]
cell = Cell(atoms=atoms, lattice_vectors=np.eye(3) * 30.0, pbc_enabled=True)

# 准备测试距离（包含更多近距离点以便观察）
distances = np.linspace(2.0, 6.0, 1000)
forces = []
energies = []

# 修改原子位置，计算不同距离下的力和能量
for d in distances:
    cell.atoms[1].position = np.array([d, 0.0, 0.0])

    # 计算力和能量
    eam_potential.calculate_forces(cell)
    forces.append(cell.atoms[0].force[0])  # 只记录x方向力
    energies.append(eam_potential.calculate_energy(cell))

# 绘制力-距离和能量-距离曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(distances, forces, label="EAM Force")
plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
plt.axvline(x=2.322, color="r", linestyle="--", alpha=0.3, label="Expected equilibrium")
plt.title("Force vs. Distance")
plt.xlabel("Distance (Å)")
plt.ylabel("Force (eV/Å)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(distances, energies, label="EAM Energy")
plt.axvline(x=2.322, color="r", linestyle="--", alpha=0.3, label="Expected equilibrium")
plt.title("Energy vs. Distance")
plt.xlabel("Distance (Å)")
plt.ylabel("Energy (eV)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
