import numpy as np
import matplotlib.pyplot as plt
from python.structure import Cell, Atom
from python.potentials import EAMAl1Potential


def create_two_atom_cell(distance, box_size=20.0):
    """
    创建包含两个原子的模拟盒子
    """
    lattice_vectors = np.eye(3) * box_size
    atoms = [
        Atom(id=0, symbol="Al", mass_amu=26.98, position=np.array([0.0, 0.0, 0.0])),
        Atom(
            id=1, symbol="Al", mass_amu=26.98, position=np.array([distance, 0.0, 0.0])
        ),
    ]
    return Cell(lattice_vectors=lattice_vectors, atoms=atoms)


# 创建EAM势实例
potential = EAMAl1Potential()


# 计算能量函数
def calculate_energy(distance):
    """使用CPP接口计算两个原子之间的能量"""
    cell = create_two_atom_cell(distance)
    return potential.calculate_energy(cell)


# 使用中心差分计算力
def calculate_force(distance, dr=1e-6):
    """使用中心差分计算力"""
    return -(calculate_energy(distance + dr) - calculate_energy(distance - dr)) / (
        2 * dr
    )


# 创建距离数组并计算能量和力
distances = np.linspace(2.0, 6.0, 100)
energies = []
forces = []

print("Calculating energies and forces...")
for d in distances:
    energy = calculate_energy(d)
    force = calculate_force(d)
    energies.append(energy)
    forces.append(force)
    if d % 1.0 < 0.1:  # 每隔约1Å打印一次
        print(f"d = {d:.3f} A: E = {energy:.6f} eV, F = {force:.6f} eV/A")

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 能量图
ax1.plot(distances, energies, "b-", label="EAM Energy")
ax1.set_xlabel("Interatomic Distance (A)")
ax1.set_ylabel("Energy (eV)")
ax1.set_title("Two-Atom Energy vs Distance")
ax1.grid(True, alpha=0.3)

# 找到并标记最小能量点
min_idx = np.argmin(energies)
min_distance = distances[min_idx]
min_energy = energies[min_idx]
ax1.plot(min_distance, min_energy, "ro", label=f"Min at {min_distance:.3f} A")
ax1.legend()

# 力图
ax2.plot(distances, forces, "g-", label="Force (Central Diff)")
ax2.set_xlabel("Interatomic Distance (A)")
ax2.set_ylabel("Force (eV/A)")
ax2.set_title("Two-Atom Force vs Distance")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

# 打印最小能量构型的详细信息
print("\nEquilibrium configuration:")
print(f"Distance: {min_distance:.6f} A")
print(f"Energy: {min_energy:.6f} eV")
print(f"Force at equilibrium: {forces[min_idx]:.6e} eV/A")

# 检查一些关键距离的值
key_distances = [2.0, 2.5, 3.0, 3.5, 4.0]
print("\nValues at key distances:")
print("Distance (A) | Energy (eV) | Force (eV/A)")
print("-" * 45)
for d in key_distances:
    idx = np.abs(distances - d).argmin()
    print(f"{distances[idx]:11.3f} | {energies[idx]:10.3f} | {forces[idx]:11.3f}")
