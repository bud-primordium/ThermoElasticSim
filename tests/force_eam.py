import numpy as np
import matplotlib.pyplot as plt
from python.structure import Cell, Atom
from python.potentials import EAMAl1Potential


def create_two_atom_cell(distance, box_size=50.0):
    """创建包含两个原子的模拟盒子"""
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


# 直接计算力函数
def calculate_force_direct(distance):
    """使用CPP接口直接计算力"""
    cell = create_two_atom_cell(distance)
    potential.calculate_forces(cell)
    return cell.atoms[0].force[0]  # 取第一个原子的x方向分量


# 中心差分计算力函数
def calculate_force_diff(distance, dr=1e-6):
    """使用中心差分计算力"""
    return -(
        potential.calculate_energy(create_two_atom_cell(distance + dr))
        - potential.calculate_energy(create_two_atom_cell(distance - dr))
    ) / (2 * dr)


# 计算力
distances = np.linspace(2.0, 6.0, 400)
forces_direct = []
forces_diff = []

print("Calculating forces...")
for d in distances:
    f_direct = calculate_force_direct(d)
    f_diff = calculate_force_diff(d)
    forces_direct.append(f_direct)
    forces_diff.append(f_diff)
    if d % 1.0 < 0.025:  # 每隔约1Å打印一次
        print(f"d = {d:.3f} A:")
        print(f"  Direct force: {f_direct:.6f} eV/A")
        print(f"  Diff force:   {f_diff:.6f} eV/A")

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(distances, forces_direct, "b-", label="Direct force calculation")
plt.plot(distances, forces_diff, "r--", label="Central difference")
plt.xlabel("Distance (A)")
plt.ylabel("Force (eV/A)")
plt.title("Al1 EAM Force Comparison (CPP Interface)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# 打印关键距离的值
key_distances = np.linspace(2.0, 4.5, 50)
print("\nComparison at key distances:")
print("Distance (A) | Direct Force (eV/A) | Diff Force (eV/A)")
print("-" * 60)
for d in key_distances:
    idx = np.abs(distances - d).argmin()
    print(
        f"{distances[idx]:11.3f} | {forces_direct[idx]:17.6f} | {forces_diff[idx]:16.6f}"
    )

# 计算两种方法的最大差异
max_diff = np.max(np.abs(np.array(forces_direct) - np.array(forces_diff)))
print(f"\nMaximum difference between methods: {max_diff:.6f} eV/A")

# 找到力为零的位置
zero_force_idx = np.argmin(np.abs(forces_diff))
print(f"Equilibrium distance (from diff method): {distances[zero_force_idx]:.6f} A")
