#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 EAM 势进行严格的集成测试 (累加逻辑版本)
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
import os
from thermoelasticsim.core.structure import Atom, Cell
from thermoelasticsim.potentials.eam import EAMAl1Potential

# --- 1. 从 eam_al1.cpp 精确复现势函数及其导数 (累加逻辑) ---


def cpp_phi_cumulative(r):
    """对势函数 φ(r) 的Python复现 (累加逻辑)"""
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

    # 根据附录格式，这些区间是累加的
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


def cpp_phi_grad_cumulative(r):
    """对势函数导数 dφ/dr 的Python复现 (累加逻辑)"""
    dphi = 0.0
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

    if r < 1.5:
        return -1e10
    if 1.5 <= r <= 2.3:
        exp_term = np.exp(a[0] + a[1] * r + a[2] * r**2 + a[3] * r**3)
        dphi += (a[1] + 2.0 * a[2] * r + 3.0 * a[3] * r**2) * exp_term

    if 2.3 < r <= 3.2:
        dr = 3.2 - r
        for n in range(5):
            dphi += -(n + 4) * b[n] * dr ** (n + 3)
    if 2.3 < r <= 4.8:
        dr = 4.8 - r
        for n in range(5):
            dphi += -(n + 4) * c[n] * dr ** (n + 3)
    if 2.3 < r <= 6.5:
        dr = 6.5 - r
        for n in range(5):
            dphi += -(n + 4) * d[n] * dr ** (n + 3)
    return dphi


def cpp_psi(r):
    """电子密度贡献函数 ψ(r) 的Python复现"""
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
    """嵌入能函数 F(ρ) 的Python复现"""
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


def cpp_psi_grad(r):
    """电子密度导数 dψ/dr 的Python复现"""
    dpsi = 0.0
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
            dpsi += -4.0 * c_k[i] * (r_k[i] - r) ** 3
    return dpsi


def cpp_F_grad(rho):
    """嵌入能导数 dF/dρ 的Python复现"""
    dF = -0.5 / np.sqrt(rho)
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
            dF += (n + 4) * D_n[n] * dr ** (n + 3)
    return dF


@pytest.fixture
def eam_potential():
    """提供一个EAMAl1Potential实例"""
    return EAMAl1Potential(cutoff=6.5)


def create_two_atom_cell(dist):
    """根据给定距离创建双原子晶胞的辅助函数"""
    atom1 = Atom(id=1, symbol="Al", mass_amu=26.98, position=np.array([0, 0, 0]))
    atom2 = Atom(id=2, symbol="Al", mass_amu=26.98, position=np.array([dist, 0, 0]))
    cell_vectors = np.diag([20, 20, 20])
    return Cell(cell_vectors, [atom1, atom2])


def test_eam_energy_two_atom(eam_potential):
    """严格测试双原子体系的EAM势能计算"""
    r = 3.0
    cell = create_two_atom_cell(r)
    expected_energy = cpp_phi_cumulative(r) + 2 * cpp_F(cpp_psi(r))
    calculated_energy = eam_potential.calculate_energy(cell)
    assert calculated_energy == pytest.approx(expected_energy, rel=1e-9)


def test_eam_force_two_atom(eam_potential):
    """严格测试双原子体系的EAM力计算"""
    r = 3.0
    cell = create_two_atom_cell(r)
    d_phi = cpp_phi_grad_cumulative(r)
    d_psi = cpp_psi_grad(r)
    d_F = cpp_F_grad(cpp_psi(r))
    force_magnitude = -(d_phi + 2 * d_F * d_psi)
    expected_force_on_atom2 = np.array([force_magnitude * (-1), 0, 0])
    eam_potential.calculate_forces(cell)
    calculated_force_on_atom2 = cell.atoms[1].force
    assert np.allclose(calculated_force_on_atom2, expected_force_on_atom2, atol=1e-9)


def test_eam_potential_shape(eam_potential):
    """测试势函数在近距离为正（排斥），远距离为负（吸引）"""
    cell_close = create_two_atom_cell(1.6)
    energy_close = eam_potential.calculate_energy(cell_close)
    cell_far = create_two_atom_cell(4.0)
    energy_far = eam_potential.calculate_energy(cell_far)
    assert energy_close > 0, "在近距离处，能量应为正值（排斥）"
    assert energy_far < 0, "在较远距离处，能量应为负值（吸引）"


def test_generate_cumulative_potential_plot():
    """
    生成EAM势能曲线图(累加逻辑)并保存为文件，以便直观验证。
    """
    distances = np.linspace(1.5, 6.5, 200)
    energies = [cpp_phi_cumulative(r) + 2 * cpp_F(cpp_psi(r)) for r in distances]
    plt.figure(figsize=(10, 6))
    plt.plot(distances, energies, label="Cumulative EAM Potential")
    plt.title("Cumulative EAM Potential for Al (Mendelev 2008)")
    plt.xlabel("Interatomic Distance (Å)")
    plt.ylabel("Potential Energy (eV)")
    plt.grid(True)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.7)
    plt.legend()
    plot_filename = "eam_potential_cumulative.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"\n[INFO] 累加势能曲线图已保存至: {os.path.abspath(plot_filename)}")
    assert os.path.exists(plot_filename)
