import numpy as np

from thermoelasticsim.core import CrystallineStructureBuilder
from thermoelasticsim.potentials.tersoff import TersoffC1988Potential
from thermoelasticsim.utils.utils import EV_TO_GPA


def _diamond_cell(a: float):
    builder = CrystallineStructureBuilder()
    return builder.create_diamond("C", a, (2, 2, 2))  # 64 原子，稳定且速度可接受


def test_tersoff_c1988_equilibrium_lattice_constant_close():
    # 目标 a0 参考 ~ 3.5656 Å
    a_ref = 3.5656
    pot = TersoffC1988Potential(delta=0.0)

    # 粗扫
    a_grid = np.linspace(3.40, 3.80, 81)
    e = []
    for a in a_grid:
        cell = _diamond_cell(float(a))
        e.append(pot.calculate_energy(cell))
    a0_coarse = float(a_grid[int(np.argmin(e))])

    # 细化
    a_refine = np.linspace(a0_coarse - 0.03, a0_coarse + 0.03, 61)
    e2 = []
    for a in a_refine:
        cell = _diamond_cell(float(a))
        e2.append(pot.calculate_energy(cell))
    a0 = float(a_refine[int(np.argmin(e2))])

    # 允许 ±0.02 Å 偏差（避免不同编译器/平台差异）
    assert abs(a0 - a_ref) < 0.02, f"a0={a0} Å vs ref {a_ref} Å"


def test_tersoff_uniaxial_virial_matches_energy_derivative():
    """
    在有限小单轴应变下，对比解析三体维里应力与能量有限差分导数（σ=∂U/∂ε/V）。
    目标：二者在数值上保持高一致性（误差<0.02 GPa）。
    """
    a = 3.5656
    pot = TersoffC1988Potential(delta=0.0)
    builder = CrystallineStructureBuilder()
    base = builder.create_diamond("C", a, (2, 2, 2))

    # 施加一个中等小的单轴形变，用其邻域做能量差分
    eps = 5e-3
    F = np.eye(3)
    F[0, 0] += eps
    cell = base.copy()
    cell.apply_deformation(F)

    # 解析三体维里
    sigma = cell.calculate_stress_tensor(pot) * EV_TO_GPA
    sxx = float(sigma[0, 0])

    # 能量有限差分（仅用于测试，不依赖库内FD实现）
    de = 1e-6
    Fp = np.eye(3)
    Fp[0, 0] += de
    Fm = np.eye(3)
    Fm[0, 0] -= de
    cp = base.copy()
    cp.apply_deformation(F @ Fp)
    cm = base.copy()
    cm.apply_deformation(F @ Fm)
    Up = float(pot.calculate_energy(cp))
    Um = float(pot.calculate_energy(cm))
    dU = (Up - Um) / (2.0 * de)
    sxx_fd = dU / cell.volume * EV_TO_GPA

    assert abs(sxx - sxx_fd) < 2e-2, f"virial {sxx:.6f} vs FD {sxx_fd:.6f} (GPa)"


def _fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])


def test_diamond_c44_symmetric_shear_reasonable():
    """
    用对称剪切应变 εxy=γ/2 直接计算 τxy-γ 的斜率，得到 C44。
    与 OpenKIM 参考 641.54 GPa 比较，允许 ~8% 的相对误差。
    """
    a = 3.5656
    pot = TersoffC1988Potential(delta=0.0)
    builder = CrystallineStructureBuilder()
    base = builder.create_diamond("C", a, (2, 2, 2))

    gammas = np.array([-1e-3, 0.0, 1e-3], dtype=float)
    tau = []
    for g in gammas:
        F = np.eye(3)
        F[0, 1] += g / 2.0
        F[1, 0] += g / 2.0
        c = base.copy()
        c.apply_deformation(F)
        sig = c.calculate_stress_tensor(pot) * EV_TO_GPA
        tau.append(sig[0, 1])  # σ12

    C44 = _fit_slope(gammas, np.array(tau))
    ref = 641.54
    rel_err = abs(C44 - ref) / ref
    assert rel_err < 0.08, f"C44={C44:.2f} vs {ref:.2f} (rel {rel_err:.3%})"


def test_diamond_size_consistency_c11_222_vs_333():
    """
    比较 2×2×2 与 3×3×3 的 C11（小应变拟合），应当几乎一致（短程势，无明显尺寸效应）。
    允许相对差异 < 0.2%。
    """
    a = 3.5656
    pot = TersoffC1988Potential(delta=0.0)
    builder = CrystallineStructureBuilder()
    base2 = builder.create_diamond("C", a, (2, 2, 2))
    base3 = builder.create_diamond("C", a, (3, 3, 3))

    strains = np.array([-2e-5, 0.0, 2e-5], dtype=float)

    def compute_c11(cell):
        sxx = []
        for e in strains:
            F = np.eye(3)
            F[0, 0] += e
            c = cell.copy()
            c.apply_deformation(F)
            sig = c.calculate_stress_tensor(pot) * EV_TO_GPA
            sxx.append(sig[0, 0])
        return _fit_slope(strains, np.array(sxx))

    C11_2 = compute_c11(base2)
    C11_3 = compute_c11(base3)
    rel = abs(C11_2 - C11_3) / max(1.0, abs(C11_3))
    assert rel < 0.002, f"C11(222)={C11_2:.2f}, C11(333)={C11_3:.2f}, rel={rel:.3%}"
