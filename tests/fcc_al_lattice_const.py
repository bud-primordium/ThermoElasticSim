import numpy as np
import matplotlib.pyplot as plt

from python.structure import Cell, Atom
from python.potentials import EAMAl1Potential


def create_fcc_cell(lattice_constant):
    """
    创建基本的 FCC 单胞（4个原子）
    """
    # 创建晶格向量
    lattice_vectors = np.array(
        [
            [lattice_constant, 0, 0],
            [0, lattice_constant, 0],
            [0, 0, lattice_constant],
        ]
    )

    # 基本 FCC 的分数坐标
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 角顶点
            [0.0, 0.5, 0.5],  # 面心
            [0.5, 0.0, 0.5],  # 面心
            [0.5, 0.5, 0.0],  # 面心
        ]
    )

    # 将分数坐标转换为笛卡尔坐标
    cart_coords = np.dot(frac_coords, lattice_vectors)

    # 创建原子列表
    atoms = []
    for i, pos in enumerate(cart_coords):
        atoms.append(Atom(id=i, symbol="Al", mass_amu=26.98, position=pos))

    # 创建晶胞
    return Cell(lattice_vectors=lattice_vectors, atoms=atoms)


def create_supercell(cell, repetition):
    """
    使用 Cell 类的 build_supercell 方法创建超胞
    """
    return cell.build_supercell(repetition)


# 创建 EAM 势
potential = EAMAl1Potential()

# 定义超胞的重复次数
repetition = (4, 4, 4)  # 在 x, y, z 方向上各重复 3 次

# 扫描晶格常数
lattice_constants = np.linspace(3.9, 4.1, 200)  # A，缩小范围以提高精度
energies = []

# 计算每个晶格常数下的能量
for a in lattice_constants:
    base_cell = create_fcc_cell(a)
    super_cell = create_supercell(base_cell, repetition)
    energy = potential.calculate_energy(super_cell)
    # 计算每个原子的平均能量
    energy_per_atom = energy / super_cell.num_atoms
    energies.append(energy_per_atom)
    print(f"a = {a:.3f} A, E = {energy_per_atom:.6f} eV/atom")

# 找到最小能量点
min_idx = np.argmin(energies)
optimal_a = lattice_constants[min_idx]
min_energy = energies[min_idx]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(
    lattice_constants,
    energies,
    "b-",
    linewidth=2,
    label="Energy per Atom vs. Lattice Constant",
)
plt.scatter(
    [optimal_a],
    [min_energy],
    color="red",
    s=100,
    label=f"Minimum at {optimal_a:.3f} Å",
)

plt.xlabel(r"Lattice Constant (Å)")
plt.ylabel("Energy per Atom (eV)")
plt.title("FCC Al Lattice Constant Optimization with Supercell")
plt.text(
    0.95,
    0.05,
    f"Supercell repetition: {repetition}\nTotal atoms: {np.prod(repetition) * 4}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.5
    ),
)
plt.grid(True, alpha=0.3)
plt.legend()

# 添加最小点的标注
plt.annotate(
    f"Minimum Energy per Atom: {min_energy:.6f} eV",
    xy=(optimal_a, min_energy),
    xytext=(optimal_a + 0.005, min_energy + 0.005),
    textcoords="data",
    ha="left",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
    arrowprops=dict(arrowstyle="->"),
)

plt.tight_layout()
plt.savefig("fcc_al_lattice_optimization.png")
plt.show()

print(f"\nOptimal lattice constant: {optimal_a:.3f} Å")
print(f"Minimum energy per atom: {min_energy:.6f} eV")

# 计算平衡构型下的力
equilibrium_base_cell = create_fcc_cell(optimal_a)
equilibrium_super_cell = create_supercell(equilibrium_base_cell, repetition)
potential.calculate_forces(equilibrium_super_cell)

# 检查力的大小
forces = np.array([atom.force for atom in equilibrium_super_cell.atoms])
max_force = np.max(np.abs(forces))
print(f"Maximum force component in equilibrium: {max_force:.6e} eV/Å")
