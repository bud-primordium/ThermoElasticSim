#!/usr/bin/env python3
"""
对比 EAM 和 Lennard-Jones 势能曲线
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# --- 1. 定义势能函数 ---


# Lennard-Jones 势
def lj_potential(r, epsilon, sigma):
    """计算给定距离r处的LJ势能"""
    if r == 0:
        return float("inf")
    sr6 = (sigma / r) ** 6
    sr12 = sr6**2
    return 4 * epsilon * (sr12 - sr6)


# 从 test_eam.py 复制过来的、经过验证的EAM势函数 (累加逻辑)
def cpp_phi_cumulative(r):
    phi_val = 0.0
    a = [0.65196946237834, 7.6046051582736, -5.8187505542843, 1.0326940511805]
    b = [
        13.695567100510,
        -44.514029786506,
        95.853674731436,
        -83.744769235189,
        29.906639687889,
    ]
    c = [
        -2.3612121457801,
        2.5279092055084,
        -3.3656803584012,
        0.94831589893263,
        -0.20965407907747,
    ]
    d = [
        0.24809459274509,
        -0.54072248340384,
        0.46579408228733,
        -0.18481649031556,
        0.028257788274378,
    ]
    if 1.5 <= r <= 2.3:
        phi_val += np.exp(a[0] + a[1] * r + a[2] * r**2 + a[3] * r**3)
    if 2.3 < r <= 3.2:
        dr = 3.2 - r
        for n in range(5):
            phi_val += b[n] * dr ** (n + 4)
    if 2.3 < r <= 4.8:
        dr = 4.8 - r
        for n in range(5):
            phi_val += c[n] * dr ** (n + 4)
    if 2.3 < r <= 6.5:
        dr = 6.5 - r
        for n in range(5):
            phi_val += d[n] * dr ** (n + 4)
    return phi_val


def cpp_psi(r):
    psi_val = 0.0
    c_k = [
        0.00019850823042883,
        0.10046665347629,
        0.10054338881951,
        0.099104582963213,
        0.090086286376778,
        0.0073022698419468,
        0.014583614223199,
        -0.0010327381407070,
        0.0073219994475288,
        0.0095726042919017,
    ]
    r_k = [2.5, 2.6, 2.7, 2.8, 3.0, 3.4, 4.2, 4.8, 5.6, 6.5]
    for i in range(10):
        if r <= r_k[i]:
            psi_val += c_k[i] * (r_k[i] - r) ** 4
    return psi_val


def cpp_F(rho):
    F_val = -np.sqrt(rho)
    if rho >= 16.0:
        dr = rho - 16.0
        D_n = [
            -6.1596236428225e-5,
            1.4856817073764e-5,
            -1.4585661621587e-6,
            7.2242013524147e-8,
            -1.7925388537626e-9,
            1.7720686711226e-11,
        ]
        for n in range(6):
            F_val += D_n[n] * dr ** (n + 4)
    return F_val


def eam_potential_total(r):
    """计算双原子体系的总EAM势能"""
    return cpp_phi_cumulative(r) + 2 * cpp_F(cpp_psi(r))


# --- 2. 绘图 ---


def test_plot_potential_comparison():
    """
    生成 EAM 和 LJ 势能的对比图。
    这个“测试”主要用于生成工件，总是会通过。
    """
    # LJ 参数 (来自CHARMM力场)
    lj_epsilon_al = 0.1743  # eV
    lj_sigma_al = 2.6059  # Angstrom

    # 距离范围 (从1.8Å开始)
    distances = np.linspace(1.8, 6.5, 300)

    # 计算能量
    eam_energies = [eam_potential_total(r) for r in distances]
    lj_energies = [lj_potential(r, lj_epsilon_al, lj_sigma_al) for r in distances]

    # 绘图
    plt.figure(figsize=(12, 7))
    plt.plot(
        distances,
        eam_energies,
        label="EAM Potential (Cumulative)",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        distances,
        lj_energies,
        label=f"L-J Potential (Al, ε={lj_epsilon_al:.4f}, σ={lj_sigma_al:.4f})",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    plt.title("Comparison of EAM and Lennard-Jones Potentials for Al")
    plt.xlabel("Interatomic Distance (Å)")
    plt.ylabel("Potential Energy (eV)")
    plt.grid(True)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
    plt.legend()

    # 自动调整y轴范围
    min_energy = min(min(eam_energies), min(lj_energies))
    max_energy = max(max(eam_energies), max(lj_energies))
    plt.ylim(min_energy - 0.1 * abs(min_energy), max_energy + 0.1 * abs(max_energy))

    plot_filename = "eam_vs_lj_potential_comparison.png"
    plt.savefig(plot_filename)
    plt.close()

    print(f"\n[INFO] 势能对比图已保存至: {os.path.abspath(plot_filename)}")
    assert os.path.exists(plot_filename)
