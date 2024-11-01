import numpy as np
import matplotlib.pyplot as plt
import logging

# 设置 matplotlib 的日志级别以抑制字体调试信息
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# 假设您有一个名为 'python' 的包，其中包含 'structure' 和 'potentials' 模块
# 请确保这些模块中包含必要的类和函数
from python.structure import Cell, Atom
from python.potentials import EAMAl1Potential


def create_fcc_supercell(lattice_constant, reps=(1, 1, 1)):
    """
    创建包含多个基本 FCC 单胞的超级晶胞。

    :param lattice_constant: 晶格常数（Å）
    :param reps: 在每个方向上的复制次数，如 (2, 2, 2)
    :return: Cell 对象
    """
    nx, ny, nz = reps
    # 创建晶格向量
    a = lattice_constant
    lattice_vectors = np.array([[a * nx, 0, 0], [0, a * ny, 0], [0, 0, a * nz]])

    # 基本 FCC 的分数坐标
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 角顶点
            [0.0, 0.5, 0.5],  # 面心
            [0.5, 0.0, 0.5],  # 面心
            [0.5, 0.5, 0.0],  # 面心
        ]
    )

    # 生成所有复制单元的原子坐标
    atoms = []
    atom_id = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k])
                for frac in frac_coords:
                    cart = (origin + frac) * lattice_constant
                    atoms.append(
                        Atom(id=atom_id, symbol="Al", mass_amu=26.98, position=cart)
                    )
                    atom_id += 1

    # 创建晶胞
    return Cell(lattice_vectors=lattice_vectors, atoms=atoms)


def check_atomic_distances(cell, threshold=1.0):
    """
    检查晶胞中所有原子对的距离，确保没有原子重叠。

    :param cell: Cell 对象
    :param threshold: 距离阈值（Å），低于此值将触发警告
    """
    positions = np.array([atom.position for atom in cell.atoms])
    num_atoms = len(positions)
    lattice_vectors = cell.lattice_vectors
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            rij = positions[j] - positions[i]
            # 应用最小镜像原则
            rij -= lattice_vectors * np.round(rij / lattice_vectors)
            distance = np.linalg.norm(rij)
            if distance < threshold:
                print(f"警告: 原子 {i} 和 原子 {j} 距离过近: {distance:.3f} Å")


# 创建 EAM 势
potential = EAMAl1Potential()

# 扫描晶格常数
lattice_constants = np.linspace(3.8, 4.3, 50)  # Å
energies = []

# 选择超级晶胞的复制次数（例如 2x2x2）
supercell_reps = (2, 2, 2)  # 可以根据需要调整

# 计算每个晶格常数下的能量
for a in lattice_constants:
    cell = create_fcc_supercell(a, reps=supercell_reps)
    check_atomic_distances(cell)  # 检查原子间距
    energy = potential.calculate_energy(cell)
    energies.append(energy)
    print(f"a = {a:.3f} Å, E = {energy:.6f} eV")

# 找到最小能量点
min_idx = np.argmin(energies)
optimal_a = lattice_constants[min_idx]
min_energy = energies[min_idx]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(
    lattice_constants, energies, "b-", linewidth=2, label="Energy vs. Lattice Constant"
)
plt.scatter(
    [optimal_a], [min_energy], color="red", s=100, label=f"Minimum at {optimal_a:.3f} Å"
)

plt.xlabel(r"Lattice Constant (\AA)")
plt.ylabel("Energy (eV)")
plt.title("FCC Al Lattice Constant Optimization")
plt.grid(True, alpha=0.3)
plt.legend()

# 添加最小点的标注
plt.annotate(
    f"Minimum Energy: {min_energy:.6f} eV",
    xy=(optimal_a, min_energy),
    xytext=(optimal_a + 0.02, min_energy + 0.02),
    textcoords="data",
    ha="left",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
    arrowprops=dict(arrowstyle="->"),
)

plt.tight_layout()
plt.show()

print(f"\nOptimal lattice constant: {optimal_a:.3f} Å")
print(f"Minimum energy: {min_energy:.6f} eV")

# 计算平衡构型下的力
equilibrium_cell = create_fcc_supercell(optimal_a, reps=supercell_reps)
potential.calculate_forces(equilibrium_cell)

# 检查力的大小
forces = np.array([atom.force for atom in equilibrium_cell.atoms])
max_force = np.max(np.abs(forces))
print(f"Maximum force component in equilibrium: {max_force:.6e} eV/Å")
