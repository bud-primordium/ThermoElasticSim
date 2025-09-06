#!/usr/bin/env python3
"""
弹性波动力学模拟模块

本模块提供弹性波在晶格中传播的分子动力学模拟功能。

主要功能
--------
- 平面波激发：支持位移型和速度型激发，以及时间域源注入
- NVE传播模拟：零温条件下的短时间线性响应模拟
- 速度测量：多种算法（互相关、到达时间拟合、k-ω谱分析）
- 可视化输出：时空图（x-t图）显示位移场和包络

物理模型
--------
- 传播方向：当前仅支持[100]方向（x轴）
- 边界条件：y/z方向周期性边界，x方向准无限介质
- 激发模式：小振幅线性响应区，避免非线性效应
- 吸收边界：可选的海绵层减少边界反射

单位约定
--------
- 长度：Å（埃）
- 时间：fs（飞秒）
- 速度：Å/fs（内部），km/s（输出，换算关系：1 Å/fs = 100 km/s）

Notes
-----
本模块为教学演示设计，追求物理清晰性和代码可理解性。
后续可扩展支持更多传播方向和材料类型。
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...core import CrystallineStructureBuilder
from ...core.structure import Cell
from ...md.schemes import NVEScheme
from ...potentials.eam import EAMAl1Potential
from ...utils.plot_config import plt

Axis = Literal["x", "y", "z"]
Polarization = Literal["L", "Ty", "Tz"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WaveExcitation:
    """平面波激发器（教学版）

    Parameters
    ----------
    direction : Axis
        传播方向，当前仅支持 "x"（等价于[100]）。
    polarization : {"L", "Ty", "Tz"}
        极化：纵波 L（平行 x），或横波 Ty/Tz（沿 y/z）。
    n_waves : int
        超胞长度内包含的整波数（满足 PBC 的允许波矢）。
    amplitude_velocity : float
        初始速度幅值（Å/fs），建议 1e-4–1e-3。
    amplitude_displacement : float | None
        初始位移幅值（Å）；若提供则叠加位移型激发（可为空）。
    phase : float
        初始相位（弧度）。

    Notes
    -----
    - 平面波形式： s(r) = A cos(k·r + φ)
    - k = 2π n / Lx · ex（n 为整数，满足 PBC）
    - 纵波：极化 e = ex；横波：e = ey 或 ez。
    - 施加后移除质心平动，避免总动量漂移。
    """

    direction: Axis = "x"
    polarization: Polarization = "L"
    n_waves: int = 2
    amplitude_velocity: float = 1e-4
    amplitude_displacement: float | None = None
    phase: float = 0.0
    mode: Literal["standing", "traveling"] = "standing"
    phase_speed_km_s: float | None = None  # 仅当 mode="traveling" 时用于设定 ω = k v

    def apply(self, cell: Cell) -> None:
        """对晶胞施加平面波激发（原地修改 cell）。"""
        if self.direction != "x":
            raise NotImplementedError("MVP仅支持[100]方向（x轴）")

        # 基本参数
        Lx = float(np.linalg.norm(cell.lattice_vectors[0]))
        k = 2.0 * np.pi * float(self.n_waves) / Lx  # 仅沿 x

        # 极化单位向量
        if self.polarization == "L":
            e = np.array([1.0, 0.0, 0.0])
        elif self.polarization == "Ty":
            e = np.array([0.0, 1.0, 0.0])
        elif self.polarization == "Tz":
            e = np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError(f"未知极化: {self.polarization}")

        # 初始位置与速度
        R = cell.get_positions()  # (N,3)
        x = R[:, 0]
        phase = k * x + float(self.phase)
        c = np.cos(phase)
        s = np.sin(phase)

        # traveling: 设定 u(x,0) = A cos(kx+φ), v(x,0) = + ω A sin(kx+φ)（沿 +x 行波）
        if self.mode == "traveling":
            v_phase_km_s = self.phase_speed_km_s
            if v_phase_km_s is None or v_phase_km_s <= 0:
                raise ValueError("traveling 模式需要有效的 phase_speed_km_s（km/s）")
            v_aa_fs = v_phase_km_s / 100.0  # Å/fs
            omega = k * v_aa_fs

            # 自动匹配位移/速度幅值关系：A_v = ω A_u
            A_v = float(self.amplitude_velocity)
            A_u = self.amplitude_displacement
            if (A_u is None or A_u == 0.0) and (A_v is not None and A_v != 0.0):
                A_u = A_v / max(omega, 1e-12)
            elif (A_v is None or A_v == 0.0) and (A_u is not None and A_u != 0.0):
                A_v = omega * A_u
            elif (A_v is None or A_v == 0.0) and (A_u is None or A_u == 0.0):
                # 回退到默认速度幅值，推导位移幅值
                A_v = 1e-4
                A_u = A_v / max(omega, 1e-12)

            # 应用位移（cos）
            if A_u and A_u != 0.0:
                du = (float(A_u) * c)[:, None] * e[None, :]
                Rn = R + du
                Rn = cell.apply_periodic_boundary(Rn)
                cell.set_positions(Rn)

            # 应用速度（+ωA sin）
            if A_v and A_v != 0.0:
                dv = (float(A_v) * s)[:, None] * e[None, :]
                V = cell.get_velocities()
                V = V + dv
                for i, atom in enumerate(cell.atoms):
                    atom.velocity = V[i]

        else:
            # standing: 速度/位移均按 cos(kx+φ)
            if self.amplitude_velocity and self.amplitude_velocity != 0.0:
                dv = (self.amplitude_velocity * c)[:, None] * e[None, :]
                V = cell.get_velocities()
                V = V + dv
                for i, atom in enumerate(cell.atoms):
                    atom.velocity = V[i]

            if (
                self.amplitude_displacement is not None
                and self.amplitude_displacement != 0.0
            ):
                du = (float(self.amplitude_displacement) * c)[:, None] * e[None, :]
                Rn = R + du
                Rn = cell.apply_periodic_boundary(Rn)
                cell.set_positions(Rn)

        # 去除整体平动（数值稳健）
        cell.remove_com_motion()


@dataclass(slots=True)
class DynamicsConfig:
    """MD 波传播最小配置。"""

    supercell: tuple[int, int, int] = (64, 12, 12)
    dt_fs: float = 0.5
    steps: int = 6000  # 3 ps @ 0.5 fs（默认更轻量）
    sample_every: int = 50
    direction: Axis = "x"
    polarization: Polarization = "L"
    n_waves: int = 2
    amplitude_velocity: float = 1e-4
    amplitude_displacement: float | None = None
    # 源注入（无需解析速度）：沿x=0薄片区域在时间域施加高斯脉冲速度
    use_source: bool = True
    source_slab_fraction: float = 0.06  # 源区域厚度占Lx比例
    source_amplitude_velocity: float = 5e-4  # Å/fs
    source_t0_fs: float = 200.0  # 脉冲中心 (fs) - 仅 Gaussian 使用
    source_sigma_fs: float = 80.0  # 脉冲宽度 (fs) - 仅 Gaussian 使用
    source_type: Literal["gaussian", "tone_burst"] = "gaussian"
    source_cycles: int = 4  # tone burst 周期数
    source_freq_thz: float = 1.0  # tone burst 载频（THz）
    record_trajectory: bool = False
    trajectory_file: str | None = None
    detector_frac_a: float = 0.20
    detector_frac_b: float = 0.70
    measure_method: str = (
        "auto"  # auto | arrival | arrival_T_guard | xcorr | threshold | komega
    )
    v_max_km_s: float | None = None
    # 吸收边界（海绵层）
    absorber_enabled: bool = False
    absorber_slab_fraction: float = 0.10  # 左右各占Lx的比例
    absorber_tau_fs: float = 250.0  # 衰减时间常数（fs），越小吸收越强
    absorber_profile: Literal["cos2", "linear"] = "cos2"


def _bin_average_signal(
    cell: Cell,
    u: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """对投影信号按 x 方向分箱取 yz 平均。

    Parameters
    ----------
    cell : Cell
        晶胞
    u : ndarray, shape (N,)
        原子标量信号（如位移在极化方向的投影）
    n_bins : int
        x 方向分箱数

    Returns
    -------
    (x_centers, avg)
        箱中心坐标（Å）与分箱平均值（shape=(n_bins,)）。
    """
    R = cell.get_positions()
    Lx = float(np.linalg.norm(cell.lattice_vectors[0]))
    s = (R[:, 0] / Lx) % 1.0  # 分数坐标（x）
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(s, edges) - 1, 0, int(n_bins) - 1)
    avg = np.zeros(int(n_bins), dtype=float)
    cnt = np.zeros(int(n_bins), dtype=int)
    for i in range(cell.num_atoms):
        avg[idx[i]] += float(u[i])
        cnt[idx[i]] += 1
    cnt = np.maximum(cnt, 1)
    avg = avg / cnt
    x_centers = (0.5 * (edges[:-1] + edges[1:])) * Lx
    return x_centers, avg


def _estimate_velocity_by_xcorr(
    t_fs: np.ndarray,
    xa: float,
    xb: float,
    x_centers: np.ndarray,
    U_xt: np.ndarray,
    v_pred_km_s: float | None = None,
    max_speed_km_s: float = 10.0,
    t_gate_fs: float | None = None,
    use_bandpass: bool = True,
) -> tuple[float | None, dict]:
    """用互相关法估计群速度。

    通过计算两个空间位置的信号互相关峰值，确定波传播的时间延迟，
    进而计算群速度。

    Parameters
    ----------
    t_fs : ndarray, shape (T,)
        采样时间序列（fs）
    xa : float
        上游探针位置（Å）
    xb : float
        下游探针位置（Å）
    x_centers : ndarray, shape (X,)
        空间分箱中心坐标（Å）
    U_xt : ndarray, shape (T, X)
        时空图：位移在极化方向的投影u·e
    v_pred_km_s : float | None
        预估速度，用于限定搜索窗口（km/s）
    max_speed_km_s : float
        物理速度上限，用于设置最小滞后约束（km/s）
    t_gate_fs : float | None
        时间窗口起始点，去除源附近近场效应（fs）
    use_bandpass : bool
        是否应用带通滤波器

    Returns
    -------
    v_kms : float | None
        估计的群速度（km/s），失败时返回None
    info : dict
        调试信息，包含滞后时间、探针位置等
    """
    # 找到最近箱
    ia = int(np.argmin(np.abs(x_centers - xa)))
    ib = int(np.argmin(np.abs(x_centers - xb)))
    if ia == ib:
        return None, {"reason": "ia==ib"}

    sa = np.asarray(U_xt[:, ia], dtype=float)
    sb = np.asarray(U_xt[:, ib], dtype=float)
    # 时间窗口：去除源附近的近场/起振阶段
    if t_gate_fs is not None:
        gate_mask = t_fs >= float(t_gate_fs)
        if np.count_nonzero(gate_mask) > 4:
            sa = sa.copy()
            sb = sb.copy()
            sa[~gate_mask] = 0.0
            sb[~gate_mask] = 0.0
    # 去均值并单位化，提升互相关鲁棒性
    sa = sa - np.nanmean(sa)
    sb = sb - np.nanmean(sb)
    sa_std = float(np.nanstd(sa)) or 1.0
    sb_std = float(np.nanstd(sb)) or 1.0
    sa = sa / sa_std
    sb = sb / sb_std

    # 可选频带：围绕上游信号的主频做简单带通，压制低频漂移与高频噪声
    if use_bandpass and len(sa) >= 16:
        dt_fs = float(t_fs[1] - t_fs[0]) if len(t_fs) >= 2 else 1.0
        fs = 1.0 / dt_fs  # fs^-1
        # FFT频率
        S = np.fft.rfft(sa - np.nanmean(sa))
        freqs = np.fft.rfftfreq(len(sa), d=dt_fs)
        # 排除直流，从第2个bin开始找峰
        if len(S) > 3:
            k0 = 1
            k_peak = int(np.argmax(np.abs(S[k0:])) + k0)
            f_peak = max(freqs[k_peak], 0.0)
            if f_peak > 0.0:
                f_lo = max(0.5 * f_peak, 0.0)
                f_hi = min(1.5 * f_peak, fs * 0.45)  # 避免靠近Nyquist

                def bp(x):
                    X = np.fft.rfft(x - np.nanmean(x))
                    H = (freqs >= f_lo) & (freqs <= f_hi)
                    X_f = X * H
                    xr = np.fft.irfft(X_f, n=len(x))
                    return xr

                sa = bp(sa)
                sb = bp(sb)

    # 互相关：sb(t) 与 sa(t-τ)，只考虑非负时延
    cc = np.correlate(sb, sa, mode="full")
    T = len(sa)
    mid = len(cc) // 2
    cc_pos = cc[mid:]
    if len(cc_pos) <= 1:
        return None, {"reason": "too_short_time_series"}
    dt_fs = float(t_fs[1] - t_fs[0]) if len(t_fs) >= 2 else 1.0

    # 引导窗口：若给定预测速度，则在 [0.5τ_pred, 1.5τ_pred] 内寻找最大峰
    if v_pred_km_s is not None and v_pred_km_s > 0:
        v_pred_aa_fs = v_pred_km_s / 100.0
        dx_A = float(abs(xb - xa))
        tau_pred_fs = dx_A / max(v_pred_aa_fs, 1e-12)
        lo = int(max(1, 0.5 * tau_pred_fs / dt_fs))
        hi = int(min(len(cc_pos), 1.5 * tau_pred_fs / dt_fs))
        if hi <= lo + 1:
            lo = max(1, int(0.03 * T))
            hi = len(cc_pos)
        search = np.abs(cc_pos[lo:hi])
        if len(search) == 0:
            return None, {"reason": "empty_window"}
        lag_idx = int(lo + np.argmax(search))
    else:
        cc_use = np.abs(cc_pos.copy())
        # 排除前若干个样本，避免拾取近零滞后的伪峰；
        # 同时基于最大物理速度设置最小滞后阈值
        dx_A = float(abs(xb - xa))
        v_max_aa_fs = max_speed_km_s / 100.0
        min_lag_fs_phys = dx_A / max(v_max_aa_fs, 1e-12)
        # 使用向上取整，确保严格满足物理最小滞后约束
        import math as _math

        ignore_head = max(int(0.03 * T), int(_math.ceil(min_lag_fs_phys / dt_fs)))
        cc_use[:ignore_head] = -np.inf
        lag_idx = int(np.argmax(cc_use))

    lag_fs = lag_idx * dt_fs
    dx_A = float(abs(xb - xa))
    if lag_fs <= 0 or dx_A <= 0:
        return None, {"reason": "nonpositive lag or dx", "lag_fs": lag_fs, "dx_A": dx_A}
    v_kms = (dx_A / lag_fs) * 100.0  # Å/fs → km/s
    return v_kms, {
        "lag_fs": lag_fs,
        "dx_A": dx_A,
        "ia": ia,
        "ib": ib,
        "lag_idx": lag_idx,
    }


def _estimate_velocity_by_threshold(
    t_fs: np.ndarray,
    sa: np.ndarray,
    sb: np.ndarray,
    dx_A: float,
    smooth_window: int = 7,
    alpha: float = 0.2,
) -> tuple[float | None, dict]:
    """基于包络阈值的到达时间差估计速度。

    Parameters
    ----------
    t_fs : ndarray
        时间序列（fs）
    sa, sb : ndarray
        两个探针的时间信号
    dx_A : float
        探针间距（Å）
    smooth_window : int
        平滑窗口（奇数）
    alpha : float
        阈值比例（相对最大值）
    """
    if dx_A <= 0 or len(t_fs) < 5:
        return None, {"reason": "invalid_inputs"}
    w = max(3, int(smooth_window) | 1)  # odd

    def smooth_abs(x):
        y = np.abs(x)
        k = np.ones(w) / w
        z = np.convolve(y, k, mode="same")
        return z

    ea = smooth_abs(sa)
    eb = smooth_abs(sb)
    ta = np.argmax(ea >= (alpha * np.max(ea)))
    tb = np.argmax(eb >= (alpha * np.max(eb)))
    if tb <= ta:
        return None, {"reason": "non_positive_dt", "ta_idx": int(ta), "tb_idx": int(tb)}
    lag_fs = t_fs[min(tb, len(t_fs) - 1)] - t_fs[min(ta, len(t_fs) - 1)]
    if lag_fs <= 0:
        return None, {"reason": "nonpositive_lag"}
    v_kms = (dx_A / lag_fs) * 100.0
    return v_kms, {"ta_fs": float(t_fs[ta]), "tb_fs": float(t_fs[tb]), "alpha": alpha}


def _estimate_velocity_by_envelope_peak(
    t_fs: np.ndarray,
    sa: np.ndarray,
    sb: np.ndarray,
    dx_A: float,
    smooth_window: int = 21,
) -> tuple[float | None, dict]:
    """基于包络峰值时间差的速度估计。

    使用移动均方根近似包络：env ≈ sqrt(moving_average(s^2))。
    取各自最大值的时间差作为到达时间差。
    """
    if dx_A <= 0 or len(t_fs) < 5:
        return None, {"reason": "invalid_inputs"}

    w = max(5, int(smooth_window) | 1)

    def envelope(x):
        x2 = np.square(x)
        k = np.ones(w) / w
        m = np.convolve(x2, k, mode="same")
        return np.sqrt(np.maximum(m, 0.0))

    ea = envelope(sa)
    eb = envelope(sb)
    ta_idx = int(np.argmax(ea))
    tb_idx = int(np.argmax(eb))
    if tb_idx <= ta_idx:
        return None, {
            "reason": "non_positive_dt_env",
            "ta_idx": ta_idx,
            "tb_idx": tb_idx,
        }
    lag_fs = float(t_fs[min(tb_idx, len(t_fs) - 1)] - t_fs[min(ta_idx, len(t_fs) - 1)])
    if lag_fs <= 0:
        return None, {"reason": "nonpositive_lag_env"}
    v_kms = (dx_A / lag_fs) * 100.0
    return v_kms, {"ta_fs": float(t_fs[ta_idx]), "tb_fs": float(t_fs[tb_idx])}


def _estimate_velocity_by_arrival_fit(
    t_fs: np.ndarray,
    x_centers: np.ndarray,
    U_xt: np.ndarray,
    x_min_A: float,
    x_max_A: float,
    t_gate_fs: float | None = None,
    alpha: float = 0.3,
    smooth_window: int = 21,
    t_max_fs: float | None = None,
) -> tuple[float | None, dict]:
    """多探针到达时间直线拟合估计速度。

    对每个空间位置计算包络达到阈值的最早时间，随后做 t(x) 的线性拟合，
    斜率 b=dt/dx，速度 v=1/b（Å/fs → km/s 乘以100）。
    """
    X = np.asarray(x_centers, dtype=float)
    T = np.asarray(t_fs, dtype=float)
    U = np.asarray(U_xt, dtype=float)
    if U.shape != (len(T), len(X)):
        return None, {"reason": "shape_mismatch"}

    # 选择空间范围
    mask_x = (x_min_A <= X) & (x_max_A >= X)
    if np.count_nonzero(mask_x) < 5:
        return None, {"reason": "insufficient_x_bins"}
    Xs = X[mask_x]
    Us = U[:, mask_x]

    # 时间窗口与早期截断裁剪（避免晚期干涉抬高阈值）
    mask_t = np.ones(len(T), dtype=bool)
    if t_gate_fs is not None:
        mask_t &= float(t_gate_fs) <= T
    if t_max_fs is not None:
        mask_t &= float(t_max_fs) >= T
    if np.count_nonzero(mask_t) > 4:
        T = T[mask_t]
        Us = Us[mask_t, :]

    # 构造包络与到达时间
    w = max(5, int(smooth_window) | 1)

    def envelope(col):
        x2 = np.square(col)
        k = np.ones(w) / w
        m = np.convolve(x2, k, mode="same")
        return np.sqrt(np.maximum(m, 0.0))

    t_arr = []
    x_sel = []
    for j in range(Us.shape[1]):
        env = envelope(Us[:, j])
        # 使用早期窗内的最大值作为阈值参考，避免晚期干涉抬高阈值
        thr = alpha * (np.max(env) if len(env) == 0 else float(np.max(env)))
        idx = int(np.argmax(env >= thr))
        if env[idx] >= thr and 0 < idx < len(T) - 1:
            t_arr.append(T[idx])
            x_sel.append(Xs[j])

    if len(t_arr) < 5:
        return None, {"reason": "insufficient_arrivals"}

    x_sel = np.asarray(x_sel, dtype=float)
    t_arr = np.asarray(t_arr, dtype=float)

    # 线性拟合 t = a + b x
    A = np.vstack([np.ones_like(x_sel), x_sel]).T
    try:
        sol, *_ = np.linalg.lstsq(A, t_arr, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        if b <= 0:
            return None, {"reason": "nonpositive_slope"}
        v_kms = (1.0 / b) * 100.0
        return v_kms, {"a_fs": a, "b_fs_per_A": b, "n_points": int(len(x_sel))}
    except Exception as e:
        return None, {"reason": f"lstsq_fail: {e}"}


def _estimate_velocity_by_komega(
    t_fs: np.ndarray,
    x_centers: np.ndarray,
    U_xt: np.ndarray,
    Lx_A: float,
    t_gate_fs: float | None = None,
    time_window_fs: float = 300.0,
) -> tuple[float | None, dict]:
    """基于 k–ω 谱峰的相速度估计 v = ω/k（Å/fs→km/s）。

    简化实现：
    - 时间窗：以 t_gate 为中心取一段窗口（或从开头截取窗口）
    - k：对空间方向做FFT，取主峰索引 m→k=2π m / L
    - ω：对上游探针做时间FFT，取主峰频率 f→ω=2π f
    """
    T = np.asarray(t_fs, dtype=float)
    X = np.asarray(x_centers, dtype=float)
    U = np.asarray(U_xt, dtype=float)
    if U.shape != (len(T), len(X)):
        return None, {"reason": "shape_mismatch"}

    # 时间窗索引
    center = float(t_gate_fs) if t_gate_fs is not None else float(T[len(T) // 2])
    half = time_window_fs / 2.0
    mask_t = (center - half <= T) & (center + half >= T)
    if np.count_nonzero(mask_t) < 8:
        mask_t = slice(None)

    Uw = U[mask_t, :]
    # 空间FFT（对每个时间求谱再平均）
    Umean = np.mean(Uw, axis=0)
    SX = np.fft.rfft(Umean - np.mean(Umean))
    m = np.argmax(np.abs(SX[1:])) + 1 if len(SX) > 2 else 0
    if m <= 0:
        return None, {"reason": "no_spatial_peak"}
    k = 2.0 * np.pi * m / max(Lx_A, 1e-12)

    # 时间FFT（上游探针处）
    ia = int(np.argmin(np.abs(X - (0.2 * Lx_A))))
    sa = U[:, ia]
    # 同样时间窗
    sa_win = sa[mask_t] if not isinstance(mask_t, slice) else sa
    dt_fs = float(T[1] - T[0]) if len(T) >= 2 else 1.0
    SF = np.fft.rfft(sa_win - np.mean(sa_win))
    freqs = np.fft.rfftfreq(len(sa_win), d=dt_fs)
    p = np.argmax(np.abs(SF[1:])) + 1 if len(SF) > 2 else 0
    if p <= 0:
        return None, {"reason": "no_temporal_peak"}
    f = freqs[p]
    omega = 2.0 * np.pi * f
    if k <= 0 or omega <= 0:
        return None, {"reason": "nonpositive_k_or_omega", "k": k, "omega": omega}
    v_aa_fs = omega / k
    v_kms = v_aa_fs * 100.0
    return v_kms, {"k_Ainv": k, "omega_fs_inv": omega}


def _first_arrival_time(
    T: np.ndarray, s: np.ndarray, alpha: float = 0.25, smooth_window: int = 21
) -> float | None:
    """返回包络首次超过阈值的时间（fs），失败返回None。"""
    if len(T) < 5:
        return None
    w = max(5, int(smooth_window) | 1)
    x2 = np.square(s)
    k = np.ones(w) / w
    m = np.convolve(x2, k, mode="same")
    env = np.sqrt(np.maximum(m, 0.0))
    thr = float(alpha) * float(np.max(env) if np.max(env) > 0 else 1.0)
    idx = int(np.argmax(env >= thr))
    if env[idx] >= thr:
        return float(T[min(idx, len(T) - 1)])
    return None


def _estimate_t_velocity_with_l_constraint(
    t_fs: np.ndarray,
    x_centers: np.ndarray,
    U_T: np.ndarray,
    U_L: np.ndarray,
    x_min_A: float,
    x_max_A: float,
    t_gate_fs: float | None = None,
    alpha_T: float = 0.25,
    alpha_L: float = 0.25,
    margin_fs: float = 100.0,
    smooth_window: int = 21,
    amplitude_ratio: float = 1.2,
    t_max_fs: float | None = None,
) -> tuple[float | None, dict]:
    """横波速度估计（使用纵波到达时间约束）。

    通过纵波先到达的物理约束，剔除横波信号中的纵波成分污染，
    提高横波速度测量的准确性。

    算法流程
    ---------
    1. 空间范围限制：x∈[x_min, x_max]
    2. 对每个空间位置x：
       - 计算纵波通道的首次到达时间tL(x)
       - 通过投影去除纵波成分：U_T_res = U_T - k*U_L
       - 在残差信号上计算横波到达时间tT(x)
       - 应用物理约束：tT ≥ tL + margin_fs
    3. 线性拟合t(x)获取波速：v = 1/b (Å/fs) × 100 → km/s

    Parameters
    ----------
    t_fs : ndarray
        时间序列（fs）
    x_centers : ndarray
        空间分箱中心（Å）
    U_T : ndarray
        横波通道信号（Ty或Tz分量）
    U_L : ndarray
        纵波通道信号（x分量）
    x_min_A : float
        空间范围最小值（Å）
    x_max_A : float
        空间范围最大值（Å）
    t_gate_fs : float | None
        时间窗口起始点（fs）
    alpha_T : float
        横波阈值比例
    alpha_L : float
        纵波阈值比例
    margin_fs : float
        纵横波时间间隔约束（fs）
    smooth_window : int
        包络平滑窗口大小
    amplitude_ratio : float
        幅值比例阈值，用于过滤纵波污染严重的信号
    t_max_fs : float | None
        时间窗口截止点（fs）

    Returns
    -------
    v_kms : float | None
        估计的横波速度（km/s）
    info : dict
        包含拟合参数和数据点数
    """
    X = np.asarray(x_centers, dtype=float)
    T = np.asarray(t_fs, dtype=float)
    UT = np.asarray(U_T, dtype=float)
    UL = np.asarray(U_L, dtype=float)
    if UT.shape != (len(T), len(X)) or UL.shape != (len(T), len(X)):
        return None, {"reason": "shape_mismatch"}

    mask_x = (x_min_A <= X) & (x_max_A >= X)
    if np.count_nonzero(mask_x) < 5:
        return None, {"reason": "insufficient_x_bins"}
    Xs = X[mask_x]
    UTs = UT[:, mask_x]
    ULs = UL[:, mask_x]

    # 时间窗口与早期截断
    mask_t = np.ones(len(T), dtype=bool)
    if t_gate_fs is not None:
        mask_t &= float(t_gate_fs) <= T
    if t_max_fs is not None:
        mask_t &= float(t_max_fs) >= T
    if np.count_nonzero(mask_t) > 4:
        T = T[mask_t]
        UTs = UTs[mask_t, :]
        ULs = ULs[mask_t, :]

    t_arr_T = []
    x_sel = []
    ncols = UTs.shape[1]
    for j in range(ncols):
        colL = ULs[:, j]
        colT = UTs[:, j]
        # L 到达
        tL = _first_arrival_time(T, colL, alpha=alpha_L, smooth_window=smooth_window)
        # 去L分量
        denom = float(np.dot(colL, colL)) + 1e-12
        k = float(np.dot(colT, colL)) / denom
        colRes = colT - k * colL
        # 幅值门控：要求T残差包络峰值显著大于L通道
        # 以避免L污染导致的过早到达
        # 计算包络峰值

        def env_max(x):
            x2 = np.square(x)
            w_local = max(5, int(smooth_window) | 1)
            kk = np.ones(w_local) / w_local
            mm = np.convolve(x2, kk, mode="same")
            return float(np.sqrt(np.max(np.maximum(mm, 0.0))))

        if env_max(colRes) < amplitude_ratio * env_max(colL):
            continue
        # T 到达
        tT = _first_arrival_time(T, colRes, alpha=alpha_T, smooth_window=smooth_window)
        if tT is None:
            continue
        if tL is not None and tT < tL + float(margin_fs):
            continue
        t_arr_T.append(tT)
        x_sel.append(Xs[j])

    if len(t_arr_T) < 5:
        return None, {"reason": "insufficient_T_arrivals"}

    x_sel = np.asarray(x_sel, dtype=float)
    t_arr_T = np.asarray(t_arr_T, dtype=float)
    A = np.vstack([np.ones_like(x_sel), x_sel]).T
    try:
        sol, *_ = np.linalg.lstsq(A, t_arr_T, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        if b <= 0:
            return None, {"reason": "nonpositive_slope_T"}
        v_kms = (1.0 / b) * 100.0
        return v_kms, {"a_fs": a, "b_fs_per_A": b, "n_points": int(len(x_sel))}
    except Exception as e:
        return None, {"reason": f"lstsq_fail_T: {e}"}


def _estimate_velocity_by_multi_xcorr(
    t_fs: np.ndarray,
    x_centers: np.ndarray,
    U_xt: np.ndarray,
    xa_frac: float,
    xb_fracs: list[float],
    Lx_A: float,
    v_max_km_s: float,
    t_gate_fs: float | None = None,
    use_bandpass: bool = True,
) -> tuple[float | None, dict]:
    """多对探针的互相关飞行时间拟合。

    - 固定上游 xa=Lx*xa_frac，多个下游 xb=Lx*xb_frac
    - 每对计算 τ_i（带时间门控/最小物理滞后/频带过滤）
    - 用最小二乘拟合 Δx = v·τ（过原点），求 v
    """
    T = np.asarray(t_fs, dtype=float)
    X = np.asarray(x_centers, dtype=float)
    U = np.asarray(U_xt, dtype=float)
    if U.shape != (len(T), len(X)):
        return None, {"reason": "shape_mismatch"}

    xa = float(xa_frac) * Lx_A
    ia = int(np.argmin(np.abs(X - xa)))
    sa = U[:, ia]
    # 时间窗口过滤
    if t_gate_fs is not None:
        gate_mask = float(t_gate_fs) <= T
        if np.count_nonzero(gate_mask) > 4:
            sa = sa.copy()
            sa[~gate_mask] = 0.0

    dXs = []
    Taus = []
    for frac in xb_fracs:
        xb = float(frac) * Lx_A
        if xb <= xa:
            continue
        ib = int(np.argmin(np.abs(X - xb)))
        sb = U[:, ib]
        # 时间窗口过滤
        sb_use = sb.copy()
        if t_gate_fs is not None and np.count_nonzero(float(t_gate_fs) <= T) > 4:
            sb_use[float(t_gate_fs) > T] = 0.0
        # xcorr 估计 τ
        # 计算滞后（fs）
        tau_fs = _xcorr_lag_sa_sb(
            T,
            sa,
            sb_use,
            dx_A=(xb - xa),
            v_max_km_s=v_max_km_s,
            t_gate_fs=t_gate_fs,
            use_bandpass=use_bandpass,
        )
        if tau_fs is None or tau_fs <= 0:
            continue
        dXs.append(xb - xa)
        Taus.append(tau_fs)

    if len(Taus) < 3:
        return None, {"reason": "insufficient_pairs"}
    dXs = np.asarray(dXs)
    Taus = np.asarray(Taus)
    # 线性拟合 Δx = v·τ（过原点）
    b = float(np.dot(dXs, Taus) / np.dot(Taus, Taus)) if np.dot(Taus, Taus) > 0 else 0.0
    if b <= 0:
        return None, {"reason": "nonpositive_slope_dx_tau"}
    v_kms = b * 100.0
    return v_kms, {"pairs": int(len(Taus))}


def _xcorr_lag_sa_sb(
    t_fs: np.ndarray,
    sa: np.ndarray,
    sb: np.ndarray,
    dx_A: float,
    v_max_km_s: float,
    t_gate_fs: float | None = None,
    use_bandpass: bool = True,
) -> float | None:
    """对两条时间序列做互相关，返回正滞后峰的时间（fs）。"""
    sa = np.asarray(sa, dtype=float)
    sb = np.asarray(sb, dtype=float)
    T = np.asarray(t_fs, dtype=float)
    if len(sa) != len(sb) or len(sa) != len(T):
        return None
    # 时间窗口过滤
    if t_gate_fs is not None:
        gate_mask = float(t_gate_fs) <= T
        if np.count_nonzero(gate_mask) > 4:
            sa = sa.copy()
            sb = sb.copy()
            sa[~gate_mask] = 0.0
            sb[~gate_mask] = 0.0
    # 去均值/标准化
    sa = sa - np.nanmean(sa)
    sb = sb - np.nanmean(sb)
    sstd_a = float(np.nanstd(sa)) or 1.0
    sstd_b = float(np.nanstd(sb)) or 1.0
    sa /= sstd_a
    sb /= sstd_b
    # 带通
    if use_bandpass and len(sa) >= 16:
        dt_fs = float(T[1] - T[0]) if len(T) >= 2 else 1.0
        fs = 1.0 / dt_fs
        S = np.fft.rfft(sa)
        freqs = np.fft.rfftfreq(len(sa), d=dt_fs)
        if len(S) > 3:
            k0 = 1
            k_peak = int(np.argmax(np.abs(S[k0:])) + k0)
            f_peak = max(freqs[k_peak], 0.0)
            if f_peak > 0.0:
                f_lo = max(0.5 * f_peak, 0.0)
                f_hi = min(1.5 * f_peak, fs * 0.45)

                def bp(x):
                    X = np.fft.rfft(x)
                    H = (freqs >= f_lo) & (freqs <= f_hi)
                    Xf = X * H
                    return np.fft.irfft(Xf, n=len(x))

                sa = bp(sa)
                sb = bp(sb)
    # 相关
    cc = np.correlate(sb, sa, mode="full")
    mid = len(cc) // 2
    cc_pos = cc[mid:]
    if len(cc_pos) <= 1:
        return None
    dt_fs = float(T[1] - T[0]) if len(T) >= 2 else 1.0
    # 最小物理滞后
    v_max_aa_fs = v_max_km_s / 100.0
    min_lag_fs_phys = dx_A / max(v_max_aa_fs, 1e-12)
    import math as _math

    ignore_head = max(1, int(_math.ceil(min_lag_fs_phys / dt_fs)))
    cc_use = np.abs(cc_pos.copy())
    cc_use[:ignore_head] = -np.inf
    lag_idx = int(np.argmax(cc_use))
    lag_fs = lag_idx * dt_fs
    if lag_fs <= 0:
        return None
    return float(lag_fs)


def simulate_plane_wave_mvp(
    material_symbol: str = "Al",
    dynamics: DynamicsConfig | None = None,
    excitation: WaveExcitation | None = None,
    out_xt_path: str | None = None,
) -> dict:
    """运行最小版 MD 平面波传播并测速（返回结果字典）。

    为教学简化，当前固定使用 EAM Al1 势与 FCC 结构。后续可抽象至 CLI。
    """
    mat_symbol = material_symbol
    dyn = dynamics or DynamicsConfig()
    exc = excitation or WaveExcitation(
        direction=dyn.direction,
        polarization=dyn.polarization,
        n_waves=dyn.n_waves,
        amplitude_velocity=dyn.amplitude_velocity,
        amplitude_displacement=dyn.amplitude_displacement,
    )

    # 1) 构建超胞（FCC 对齐 x 方向）
    a_ref = 4.045  # Å（EAM Al1 参考晶格常数）
    builder = CrystallineStructureBuilder()
    cell = builder.create_fcc(mat_symbol, a_ref, dyn.supercell)

    # 2) 初始条件（可选）。默认不注入 standing/traveling，依靠源驱动
    if not dyn.use_source:
        exc.apply(cell)

    # 3) NVE 传播并采样位移投影的 x–t 图
    pot = EAMAl1Potential()
    scheme = NVEScheme()
    # 预计算初始力，确保第一步半步更新使用正确 F(t)
    with contextlib.suppress(Exception):
        pot.calculate_forces(cell)

    R0 = cell.get_positions().copy()
    pol_vec = (
        np.array([1.0, 0.0, 0.0])
        if exc.polarization == "L"
        else (
            np.array([0.0, 1.0, 0.0])
            if exc.polarization == "Ty"
            else np.array([0.0, 0.0, 1.0])
        )
    )
    n_bins = dyn.supercell[0] * 4  # x 方向分辨率（每胞4个层面）
    xs_ref, _ = _bin_average_signal(cell, np.zeros(cell.num_atoms), n_bins)

    times = []
    U_xt = []  # projection on polarization vector (legacy)
    UX_xt = []  # longitudinal (x) component
    UY_xt = []  # transverse-y component
    UZ_xt = []  # transverse-z component

    # 轨迹记录（可选）
    writer = None
    if dyn.record_trajectory:
        from thermoelasticsim.utils.trajectory import TrajectoryWriter

        traj_path = dyn.trajectory_file or "wave_trajectory.h5"
        writer = TrajectoryWriter(traj_path)
        writer.initialize(
            cell.num_atoms, n_frames_estimate=max(2, dyn.steps // dyn.sample_every + 2)
        )

    t0 = time.time()
    log_every = max(100, int(dyn.steps // 20))
    for s in range(int(dyn.steps)):
        # 时间（fs, ps）
        t_fs = s * float(dyn.dt_fs)

        # 源注入（位于 x ∈ [0, frac*Lx] 的薄片），高斯时间包络
        if dyn.use_source and dyn.source_amplitude_velocity != 0.0:
            Lx = float(np.linalg.norm(cell.lattice_vectors[0]))
            slab_x = dyn.source_slab_fraction * Lx
            R = cell.get_positions()
            mask = R[:, 0] <= slab_x
            # 注入方向按极化
            e_inj = (
                np.array([1.0, 0.0, 0.0])
                if exc.polarization == "L"
                else (
                    np.array([0.0, 1.0, 0.0])
                    if exc.polarization == "Ty"
                    else np.array([0.0, 0.0, 1.0])
                )
            )
            amp = dyn.source_amplitude_velocity
            dv_vec = None
            if dyn.source_type == "gaussian" and dyn.source_sigma_fs > 0.0:
                g = np.exp(
                    -0.5 * ((t_fs - dyn.source_t0_fs) / dyn.source_sigma_fs) ** 2
                )
                if g > 1e-8:
                    dv_vec = amp * float(g) * e_inj
            elif (
                dyn.source_type == "tone_burst"
                and dyn.source_freq_thz > 0.0
                and dyn.source_cycles > 0
            ):
                # f_fs = THz / 1000, ω = 2π f_fs
                f_fs = dyn.source_freq_thz / 1000.0
                T_burst_fs = dyn.source_cycles / max(f_fs, 1e-12)
                if 0.0 <= t_fs <= T_burst_fs:
                    # 汉宁窗包络
                    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * t_fs / T_burst_fs))
                    dv_vec = amp * w * np.sin(2.0 * np.pi * f_fs * t_fs) * e_inj
            if dv_vec is not None:
                for i, atom in enumerate(cell.atoms):
                    if mask[i]:
                        atom.velocity += dv_vec
                # 注入产生净动量，移除体系整体质心平动，避免整体飘移/拼接错觉
                cell.remove_com_motion()

        # 正常NVE推进
        scheme.step(cell, pot, float(dyn.dt_fs))
        # 吸收边界（海绵层）：在两端对速度做指数衰减，减少绕回/反射
        if (
            dyn.absorber_enabled
            and dyn.absorber_slab_fraction > 0
            and dyn.absorber_tau_fs > 0
        ):
            Lx = float(np.linalg.norm(cell.lattice_vectors[0]))
            slab = float(dyn.absorber_slab_fraction) * Lx
            if slab > 0:
                R = cell.get_positions()
                X = R[:, 0]
                w = np.zeros(cell.num_atoms, dtype=float)
                left = slab > X
                right = (Lx - slab) < X
                if np.any(left):
                    xi = (slab - X[left]) / slab  # 0→1 从内到外
                    if dyn.absorber_profile == "linear":
                        w[left] = np.clip(xi, 0.0, 1.0)
                    else:  # cos2 平滑
                        w[left] = np.square(np.sin(0.5 * np.pi * np.clip(xi, 0.0, 1.0)))
                if np.any(right):
                    xi = (X[right] - (Lx - slab)) / slab
                    if dyn.absorber_profile == "linear":
                        w[right] = np.clip(xi, 0.0, 1.0)
                    else:
                        w[right] = np.square(
                            np.sin(0.5 * np.pi * np.clip(xi, 0.0, 1.0))
                        )
                if np.any(w > 0):
                    V = cell.get_velocities()
                    # 指数衰减因子：exp(-dt/τ * w)
                    factors = np.exp(
                        -(float(dyn.dt_fs) / float(dyn.absorber_tau_fs)) * w
                    )
                    V = (V.T * factors).T
                    for i, atom in enumerate(cell.atoms):
                        atom.velocity = V[i]
                    # 防止引入净动量
                    cell.remove_com_motion()
        if s % int(dyn.sample_every) == 0:
            times.append(t_fs)
            R = cell.get_positions()
            disp = R - R0
            u_pol = disp @ pol_vec  # 位移在极化方向投影
            ux = disp[:, 0]
            uy = disp[:, 1]
            uz = disp[:, 2]
            x_centers, avg_pol = _bin_average_signal(cell, u_pol, n_bins)
            _, avg_x = _bin_average_signal(cell, ux, n_bins)
            _, avg_y = _bin_average_signal(cell, uy, n_bins)
            _, avg_z = _bin_average_signal(cell, uz, n_bins)
            U_xt.append(avg_pol)
            UX_xt.append(avg_x)
            UY_xt.append(avg_y)
            UZ_xt.append(avg_z)
            if writer is not None:
                # 以 ps 计时间
                writer.write_frame(
                    positions=R,
                    box=cell.lattice_vectors,
                    time=t_fs / 1000.0,
                    step=s,
                )
        if s and (s % log_every == 0):
            elapsed = time.time() - t0
            pct = 100.0 * s / float(dyn.steps)
            eta = elapsed * (dyn.steps - s) / s
            logger.info(
                f"MD传播 {s}/{dyn.steps} ({pct:.1f}%), 采样帧={len(times)}, 用时={elapsed:.1f}s, 预计剩余={eta:.1f}s"
            )

    t_fs_arr = np.asarray(times, dtype=float)
    U_xt_arr = np.asarray(U_xt, dtype=float)  # shape (T, X)
    UX_arr = np.asarray(UX_xt, dtype=float)
    UY_arr = np.asarray(UY_xt, dtype=float)
    UZ_arr = np.asarray(UZ_xt, dtype=float)

    # 4) 互相关测速：选用 0.25L 与 0.50L；裁剪时间窗口避免绕回与后期干涉
    Lx = float(np.linalg.norm(cell.lattice_vectors[0]))
    xa = float(dyn.detector_frac_a) * Lx
    xb = float(dyn.detector_frac_b) * Lx
    # 物理最大速度上限（用于互相关最小滞后约束）
    if dyn.v_max_km_s is not None and dyn.v_max_km_s > 0:
        v_max_km_s = float(dyn.v_max_km_s)
    else:
        v_max_km_s = 8.0 if dyn.polarization == "L" else 5.0
    # 时间窗口过滤：对高斯取 t0+2σ；对tone burst取 burst持续时间
    if dyn.use_source:
        if dyn.source_type == "gaussian":
            t_gate = dyn.source_t0_fs + 1.2 * dyn.source_sigma_fs
        else:
            f_fs = dyn.source_freq_thz / 1000.0 if dyn.source_freq_thz > 0 else 0.0
            T_burst_fs = (dyn.source_cycles / f_fs) if f_fs > 0 else 0.0
            t_gate = T_burst_fs
    else:
        t_gate = None
    t_cut_fs = min(t_fs_arr.max(), 0.8 * Lx / (v_max_km_s / 100.0))
    mask_t = t_fs_arr <= t_cut_fs
    t_win = t_fs_arr[mask_t]
    U_win = U_xt_arr[mask_t, :]
    v_kms = None
    xinfo = {}
    candidates: list[tuple[str, float, dict]] = []
    ia = int(np.argmin(np.abs(xs_ref - xa)))
    ib = int(np.argmin(np.abs(xs_ref - xb)))

    # 优先选择估计方法：Gaussian → 多探针到达拟合（早期窗）；ToneBurst → 互相关
    if dyn.measure_method in ("arrival", "arrival_T_guard") or (
        dyn.use_source and dyn.source_type == "gaussian"
    ):
        # 多点拟合（更稳健）
        x_min_A = float(dyn.detector_frac_a) * Lx
        x_max_A = float(dyn.detector_frac_b) * Lx
        # 早期截止时间：避免晚期绕回/干涉抬高阈值
        t_early_end = float(t_gate if t_gate is not None else t_win.min()) + 0.6 * (
            Lx / (v_max_km_s / 100.0)
        )
        t_early_end = min(t_early_end, t_cut_fs)
        if dyn.polarization == "L" and dyn.measure_method != "arrival_T_guard":
            v0, info0 = _estimate_velocity_by_arrival_fit(
                t_win,
                xs_ref,
                UX_arr[mask_t, :],
                x_min_A,
                x_max_A,
                t_gate_fs=t_gate,
                alpha=0.18,
                t_max_fs=t_early_end,
            )
        else:
            # 对横波：用L通道做护栏，剔除早于L到达的分量
            U_T = UY_arr if dyn.polarization == "Ty" else UZ_arr
            v0, info0 = _estimate_t_velocity_with_l_constraint(
                t_win,
                xs_ref,
                U_T[mask_t, :],
                UX_arr[mask_t, :],
                x_min_A,
                x_max_A,
                t_gate_fs=t_gate,
                alpha_T=0.20,
                alpha_L=0.20,
                margin_fs=150.0,
                t_max_fs=t_early_end,
            )
        if v0 is not None:
            candidates.append(("arrival_fit", v0, info0))

    if dyn.measure_method in ("auto", "xcorr"):
        v1, info1 = _estimate_velocity_by_xcorr(
            t_win,
            xa,
            xb,
            xs_ref,
            U_win,
            v_pred_km_s=None,
            max_speed_km_s=v_max_km_s,
            t_gate_fs=t_gate,
            use_bandpass=True,
        )
        if v1 is not None:
            candidates.append(("xcorr", v1, info1))

    # multi_xcorr（多对探针）
    if dyn.measure_method in ("auto", "multi_xcorr", "arrival", "arrival_T_guard"):
        xb_fracs = [float(dyn.detector_frac_a) + 0.05 * i for i in range(2, 9)]
        xb_fracs = [f for f in xb_fracs if f < float(dyn.detector_frac_b)]
        v5, info5 = _estimate_velocity_by_multi_xcorr(
            t_win,
            xs_ref,
            U_win,
            xa_frac=float(dyn.detector_frac_a),
            xb_fracs=xb_fracs,
            Lx_A=Lx,
            v_max_km_s=v_max_km_s,
            t_gate_fs=t_gate,
            use_bandpass=True,
        )
        if v5 is not None:
            candidates.append(("multi_xcorr", v5, info5))

    # 如不在物理合理范围，尝试阈值法回退
    if dyn.measure_method in ("auto", "threshold"):
        v2, info2 = _estimate_velocity_by_threshold(
            t_win, U_win[:, ia], U_win[:, ib], abs(xb - xa)
        )
        if v2 is not None:
            candidates.append(("threshold", v2, info2))
    # 如仍不合理，使用包络峰值回退
    if dyn.measure_method in ("auto", "envelope"):
        v3, info3 = _estimate_velocity_by_envelope_peak(
            t_win, U_win[:, ia], U_win[:, ib], abs(xb - xa)
        )
        if v3 is not None:
            candidates.append(("envelope_peak", v3, info3))

    # k–ω 候选
    if dyn.measure_method in ("auto", "komega"):
        v4, info4 = _estimate_velocity_by_komega(
            t_win, xs_ref, U_win, Lx, t_gate_fs=t_gate
        )
        if v4 is not None:
            candidates.append(("komega", v4, info4))

    # 选择候选：
    # - 若为 Gaussian 源：优先 arrival_fit（L: 直接；T: 带 L 护栏）
    # - 若用户显式 method=arrival/arrival_T_guard：强制使用 arrival_fit
    # - 否则：优先 multi_xcorr；再回退到在物理范围内的中位数；再回退到最接近 5 km/s
    method_requested = str(dyn.measure_method or "auto")
    # 整理候选
    cand_map: dict[str, tuple[float, dict]] = {}
    for m, v, info in candidates:
        # 仅保留每种方法的一个代表值（首个）
        if m not in cand_map:
            cand_map[m] = (float(v), info)

    def _in_range(v: float) -> bool:
        return 1.0 <= float(v) <= float(v_max_km_s)

    # 1) 用户强制 arrival/arrival_T_guard
    if method_requested in ("arrival", "arrival_T_guard"):
        if "arrival_fit" in cand_map and _in_range(cand_map["arrival_fit"][0]):
            v, info = cand_map["arrival_fit"]
            v_kms, xinfo = v, {"method": "arrival_fit", **info}
        elif "multi_xcorr" in cand_map and _in_range(cand_map["multi_xcorr"][0]):
            v, info = cand_map["multi_xcorr"]
            v_kms, xinfo = v, {"method": "multi_xcorr", **info}
    # 2) Gaussian 源：尽量使用 arrival_fit
    if (
        v_kms is None
        and dyn.use_source
        and dyn.source_type == "gaussian"
        and "arrival_fit" in cand_map
    ):
        v, info = cand_map["arrival_fit"]
        if _in_range(v):
            v_kms, xinfo = v, {"method": "arrival_fit", **info}

    # 3) 优先 multi_xcorr
    if (
        v_kms is None
        and "multi_xcorr" in cand_map
        and _in_range(cand_map["multi_xcorr"][0])
    ):
        v, info = cand_map["multi_xcorr"]
        v_kms, xinfo = v, {"method": "multi_xcorr", **info}

    # 4) 在物理范围内的中位数
    if v_kms is None:
        valid = [(m, v, info) for (m, v, info) in candidates if _in_range(v)]
        if valid:
            vs = sorted([v for (_, v, __) in valid])
            v_pick = vs[len(vs) // 2]
            for m, v, info in valid:
                if abs(v - v_pick) < 1e-9:
                    v_kms, xinfo = v, {"method": m, **info}
                    break
        elif candidates:
            # 选择最接近 5 km/s 的
            m, v, info = min(candidates, key=lambda x: abs(x[1] - 5.0))
            v_kms, xinfo = v, {"method": m, **info}

    # 5) x-t 双面板图：左侧显示位移场，右侧显示包络（便于识别波前）
    if out_xt_path:
        # 尝试创建增强版4子图，失败则回退到简单版2子图
        enhanced_plot_success = False
        try:
            # 创建更复杂的布局：2x2子图
            fig = plt.figure(figsize=(12, 8))

            # 左上：位移场
            ax1 = plt.subplot(2, 2, 1)
            extent = [
                xs_ref.min(),
                xs_ref.max(),
                t_fs_arr.min() / 1000.0,
                t_cut_fs / 1000.0,
            ]

            # 位移场（对称色标）
            vmax = float(np.nanmax(np.abs(U_win))) or 1e-12
            vlim = 0.85 * vmax
            im1 = ax1.imshow(
                U_win,
                origin="lower",
                aspect="auto",
                extent=[extent[0], extent[1], extent[2], extent[3]],
                cmap="RdBu_r",
                vmin=-vlim,
                vmax=+vlim,
            )
            ax1.set_xlabel("位置 x (Å)")
            ax1.set_ylabel("时间 t (ps)")
            ax1.set_title("位移场 u·e")
            cbar1 = fig.colorbar(im1, ax=ax1)
            cbar1.set_label("位移 (Å)", rotation=270, labelpad=15)

            # 标记探测点位置
            ax1.axvline(
                x=xa,
                color="k",
                lw=0.8,
                alpha=0.5,
                linestyle="--",
                label=f"探测点A: {xa:.1f}Å",
            )
            ax1.axvline(
                x=xb,
                color="k",
                lw=0.8,
                alpha=0.5,
                linestyle=":",
                label=f"探测点B: {xb:.1f}Å",
            )

            # 右上：包络（RMS）
            ax2 = plt.subplot(2, 2, 2)
            w_env = max(5, int(len(t_win) // 20) | 1)
            K = np.ones(w_env) / w_env
            U2 = U_win * U_win
            # 沿时间轴卷积
            env = np.apply_along_axis(
                lambda col: np.convolve(col, K, mode="same"), axis=0, arr=U2
            )
            env = np.sqrt(np.maximum(env, 0.0))
            vmax_env = float(np.nanpercentile(env, 99.0)) or 1e-12

            im2 = ax2.imshow(
                env,
                origin="lower",
                aspect="auto",
                extent=[extent[0], extent[1], extent[2], extent[3]],
                cmap="hot",
                vmin=0.0,
                vmax=vmax_env,
            )
            ax2.set_xlabel("位置 x (Å)")
            ax2.set_ylabel("时间 t (ps)")
            ax2.set_title("包络 |u·e|")
            cbar2 = fig.colorbar(im2, ax=ax2)
            cbar2.set_label("包络幅度 (Å)", rotation=270, labelpad=15)

            # 标记探测点
            ax2.axvline(x=xa, color="w", lw=0.8, alpha=0.5, linestyle="--")
            ax2.axvline(x=xb, color="w", lw=0.8, alpha=0.5, linestyle=":")

            # 如果有到达拟合结果，添加波前线
            if isinstance(xinfo, dict) and ("a_fs" in xinfo and "b_fs_per_A" in xinfo):
                xs_line = np.linspace(xs_ref.min(), xs_ref.max(), 200)
                t_line_ps = (
                    float(xinfo["a_fs"]) + float(xinfo["b_fs_per_A"]) * xs_line
                ) / 1000.0
                ax1.plot(
                    xs_line,
                    t_line_ps,
                    "g--",
                    lw=1.5,
                    alpha=0.7,
                    label=f"波前 (v={v_kms:.2f} km/s)",
                )
                ax2.plot(xs_line, t_line_ps, "y--", lw=1.5, alpha=0.7)
                ax1.legend(loc="upper right", fontsize=8)

            # 左下：探测点A的时域信号
            ax3 = plt.subplot(2, 2, 3)
            ia_plot = int(np.argmin(np.abs(xs_ref - xa)))
            signal_a = U_win[:, ia_plot]
            ax3.plot(t_win / 1000.0, signal_a, "b-", lw=1.0, label=f"x={xa:.1f}Å")
            ax3.set_xlabel("时间 t (ps)")
            ax3.set_ylabel("位移 u·e (Å)")
            ax3.set_title("探测点A信号")
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # 右下：探测点B的时域信号
            ax4 = plt.subplot(2, 2, 4)
            ib_plot = int(np.argmin(np.abs(xs_ref - xb)))
            signal_b = U_win[:, ib_plot]
            ax4.plot(t_win / 1000.0, signal_b, "r-", lw=1.0, label=f"x={xb:.1f}Å")
            ax4.set_xlabel("时间 t (ps)")
            ax4.set_ylabel("位移 u·e (Å)")
            ax4.set_title("探测点B信号")
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            # 添加互相关信息（如果有）
            if v_kms is not None and "lag_fs" in xinfo:
                lag_ps = xinfo.get("lag_fs", 0) / 1000.0
                ax4.axvline(
                    x=lag_ps,
                    color="g",
                    linestyle="--",
                    alpha=0.5,
                    label=f"延迟: {lag_ps:.2f} ps",
                )
                ax4.legend()

            # 总标题
            method_str = (
                xinfo.get("method", "unknown") if isinstance(xinfo, dict) else "unknown"
            )
            title = (
                f"弹性波传播分析 - {dyn.polarization}波 | v≈{v_kms:.2f} km/s (方法: {method_str})"
                if v_kms is not None
                else f"弹性波传播分析 - {dyn.polarization}波 | 速度估计失败"
            )
            fig.suptitle(title, fontsize=12, y=0.98)

            fig.tight_layout()
            fig.savefig(out_xt_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            enhanced_plot_success = True
        except Exception:
            enhanced_plot_success = False

        # 如果增强版失败，使用简单版
        if not enhanced_plot_success:
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
            extent = [
                xs_ref.min(),
                xs_ref.max(),
                t_fs_arr.min() / 1000.0,
                t_cut_fs / 1000.0,
            ]

            # 左：位移（对称色标）
            vmax = float(np.nanmax(np.abs(U_win))) or 1e-12
            vlim = 0.85 * vmax
            im0 = axes[0].imshow(
                U_win,
                origin="lower",
                aspect="auto",
                extent=[extent[0], extent[1], extent[2], extent[3]],
                cmap="RdBu_r",
                vmin=-vlim,
                vmax=+vlim,
            )
            axes[0].set_xlabel("x (Å)")
            axes[0].set_ylabel("time (ps)")
            cbar0 = fig.colorbar(im0, ax=axes[0])
            cbar0.set_label("u·e (Å)")
            axes[0].axvline(x=xa, color="k", lw=0.8, alpha=0.35)
            axes[0].axvline(x=xb, color="k", lw=0.8, alpha=0.35)

            # 右：包络（RMS）
            w_env = max(5, int(len(t_win) // 20) | 1)
            K = np.ones(w_env) / w_env
            U2 = U_win * U_win
            # 沿时间轴卷积
            env = np.apply_along_axis(
                lambda col: np.convolve(col, K, mode="same"), axis=0, arr=U2
            )
            env = np.sqrt(np.maximum(env, 0.0))
            vmax_env = float(np.nanpercentile(env, 99.0)) or 1e-12
            im1 = axes[1].imshow(
                env,
                origin="lower",
                aspect="auto",
                extent=[extent[0], extent[1], extent[2], extent[3]],
                cmap="magma",
                vmin=0.0,
                vmax=vmax_env,
            )
            axes[1].set_xlabel("x (Å)")
            cbar1 = fig.colorbar(im1, ax=axes[1])
            cbar1.set_label("|u·e| (RMS Å)")
            axes[1].axvline(x=xa, color="w", lw=0.8, alpha=0.35)
            axes[1].axvline(x=xb, color="w", lw=0.8, alpha=0.35)

            # 若有到达拟合直线，同步叠加在两幅图上
            if isinstance(xinfo, dict) and ("a_fs" in xinfo and "b_fs_per_A" in xinfo):
                xs_line = np.linspace(xs_ref.min(), xs_ref.max(), 200)
                t_line_ps = (
                    float(xinfo["a_fs"]) + float(xinfo["b_fs_per_A"]) * xs_line
                ) / 1000.0
                for ax in axes:
                    ax.plot(
                        xs_line,
                        t_line_ps,
                        "w--" if ax is axes[1] else "k--",
                        lw=1.0,
                        alpha=0.8,
                    )

            title = (
                f"弹性波传播时空图（v≈{v_kms:.2f} km/s）"
                if v_kms is not None
                else "弹性波传播时空图（速度估计失败）"
            )
            fig.suptitle(title, y=1.02)
            fig.tight_layout()
            try:
                fig.savefig(out_xt_path, dpi=300, bbox_inches="tight")
            finally:
                plt.close(fig)

    result = {
        "material": mat_symbol,
        "supercell": dyn.supercell,
        "dt_fs": dyn.dt_fs,
        "steps": dyn.steps,
        "sample_every": dyn.sample_every,
        "direction": dyn.direction,
        "polarization": dyn.polarization,
        "n_waves": dyn.n_waves,
        "amplitude_velocity": dyn.amplitude_velocity,
        "amplitude_displacement": dyn.amplitude_displacement,
        "use_source": dyn.use_source,
        "record_trajectory_requested": bool(dyn.record_trajectory),
        "source": {
            "slab_fraction": dyn.source_slab_fraction,
            "amplitude_velocity": dyn.source_amplitude_velocity,
            "t0_fs": dyn.source_t0_fs,
            "sigma_fs": dyn.source_sigma_fs,
        },
        "x_centers_A": xs_ref.tolist(),
        "t_fs": t_fs_arr.tolist(),
        "velocity_estimate_km_s": v_kms,
        "xcorr_info": xinfo,
    }

    if writer is not None:
        # 写入元数据，便于GIF叠加信息
        with contextlib.suppress(Exception):
            writer.write_metadata(
                {
                    "polarization": str(dyn.polarization),
                    "source_type": str(dyn.source_type) if dyn.use_source else "none",
                    "detector_frac_a": float(dyn.detector_frac_a),
                    "detector_frac_b": float(dyn.detector_frac_b),
                    "dt_fs": float(dyn.dt_fs),
                    "steps": int(dyn.steps),
                    "supercell_x": int(dyn.supercell[0]),
                    "supercell_y": int(dyn.supercell[1]),
                    "supercell_z": int(dyn.supercell[2]),
                    "velocity_estimate_km_s": float(v_kms)
                    if v_kms is not None
                    else -1.0,
                    "velocity_method": str(xinfo.get("method", "unknown"))
                    if isinstance(xinfo, dict)
                    else "unknown",
                }
            )
        writer.close()
        result["trajectory_file"] = writer.filename.as_posix()

    return result


__all__ = [
    "WaveExcitation",
    "DynamicsConfig",
    "simulate_plane_wave_mvp",
]
