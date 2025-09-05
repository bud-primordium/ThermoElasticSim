#!/usr/bin/env python3
"""
弹性波解析计算模块

本模块实现立方晶系材料中，基于弹性常数 ``C11, C12, C44`` 与
材料密度 ``ρ`` 的弹性波速解析计算（阶段A）。支持任意传播方向
的Christoffel方程求解，自动区分纵波与横波，并给出相应偏振向量。

单位与约定
----------
- 弹性常数输入单位：GPa
- 密度输入单位：g/cm^3
- 输出速度单位：km/s

在上述单位下，波速满足 v[km/s] = sqrt(C[GPa] / ρ[g/cm^3])。

示例
----
>>> ana = ElasticWaveAnalyzer(C11=110.0, C12=61.0, C44=33.0, density=2.70)
>>> ana.calculate_wave_velocities([1, 0, 0])["longitudinal"]
6.38...
>>> report = ana.generate_report()
>>> sorted(report.keys())
['[100]', '[110]', '[111]']
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ElasticWaveAnalyzer:
    """
    基于弹性常数的波速解析计算器（立方晶系）。

    Parameters
    ----------
    C11, C12, C44 : float
        立方晶系弹性常数，单位 GPa。
    density : float
        材料密度，单位 g/cm^3。

    Notes
    -----
    - 使用Christoffel方程 Γ·u = v^2 u 求解本征值 v^2 与本征向量 u。
    - 在当前单位系统下，Γ 以 (km/s)^2 表示，故速度可直接取平方根。
    - 纵波的判定依据：偏振方向与传播方向夹角最小（|n·u| 最大）。
    """

    C11: float
    C12: float
    C44: float
    density: float

    # ---------------------------- 公共接口 ----------------------------
    def calculate_wave_velocities(self, direction: Iterable[float]) -> dict:
        """
        计算指定方向的纵横波速度与偏振方向。

        Parameters
        ----------
        direction : Iterable[float]
            传播方向向量或米勒指数，例如 [1,0,0], [1,1,0], [1,1,1]。

        Returns
        -------
        dict
            包含下列键：

            - 'longitudinal': float，纵波速度 (km/s)
            - 'transverse1': float，横波速度1 (km/s)
            - 'transverse2': float，横波速度2 (km/s)
            - 'polarizations': dict，包含三个模式的单位偏振向量
        """
        n = _normalize_direction(direction)
        gamma = self._christoffel_matrix(n)
        # 对称矩阵，使用 eigh（保证特征值升序、向量正交）
        vals, vecs = np.linalg.eigh(gamma)

        # 根据与传播方向的对齐程度区分纵波：|n·u| 最大者为L
        dots = np.abs(vecs.T @ n)
        idx_L = int(np.argmax(dots))

        # 纵波
        vL = float(np.sqrt(max(vals[idx_L], 0.0)))
        eL = _unit(vecs[:, idx_L])

        # 横波（其余两个）
        idx_T = [i for i in range(3) if i != idx_L]
        # 将横波按速度升序稳定排序，便于测试与复现
        t_modes = sorted(
            [(float(np.sqrt(max(vals[i], 0.0))), _unit(vecs[:, i])) for i in idx_T],
            key=lambda x: x[0],
        )
        (vT1, eT1), (vT2, eT2) = t_modes

        return {
            "longitudinal": vL,
            "transverse1": vT1,
            "transverse2": vT2,
            "polarizations": {
                "longitudinal": eL.tolist(),
                "transverse1": eT1.tolist(),
                "transverse2": eT2.tolist(),
            },
        }

    def generate_report(self) -> dict[str, dict[str, float | dict]]:
        """
        生成标准方向 [100]、[110]、[111] 的波速报告。

        Returns
        -------
        dict
            形如 ``{"[100]": {...}, "[110]": {...}, "[111]": {...}}`` 的字典。
        """
        report: dict[str, dict[str, float | dict]] = {}
        for label, d in (
            ("[100]", (1.0, 0.0, 0.0)),
            ("[110]", (1.0, 1.0, 0.0)),
            ("[111]", (1.0, 1.0, 1.0)),
        ):
            report[label] = self.calculate_wave_velocities(d)
        return report

    # ---------------------------- 内部实现 ----------------------------
    def _christoffel_matrix(self, n: np.ndarray) -> np.ndarray:
        """
        构建Christoffel矩阵 Γ（单位：(km/s)^2）。

        Parameters
        ----------
        n : ndarray, shape (3,)
            单位传播方向向量。

        Returns
        -------
        ndarray, shape (3, 3)
            Christoffel矩阵。

        Notes
        -----
        对立方晶系，Γ 可写为（省略 1/ρ 因子已并入到数值中）：

        Γ_11 = C11 n1^2 + C44 (n2^2 + n3^2)
        Γ_22 = C11 n2^2 + C44 (n1^2 + n3^2)
        Γ_33 = C11 n3^2 + C44 (n1^2 + n2^2)
        Γ_12 = (C12 + C44) n1 n2 （其余对称）

        其中 C 以 GPa，ρ 以 g/cm^3 计，Γ 的元素单位为 (km/s)^2。
        """
        C11, C12, C44 = self.C11, self.C12, self.C44
        rho = self.density
        n1, n2, n3 = float(n[0]), float(n[1]), float(n[2])

        factor = 1.0 / rho  # 将 1/ρ 吸收入系数，GPa→(km/s)^2 的一致性见模块说明

        g11 = (C11 * n1 * n1 + C44 * (n2 * n2 + n3 * n3)) * factor
        g22 = (C11 * n2 * n2 + C44 * (n1 * n1 + n3 * n3)) * factor
        g33 = (C11 * n3 * n3 + C44 * (n1 * n1 + n2 * n2)) * factor
        c12p = (C12 + C44) * factor
        g12 = c12p * n1 * n2
        g13 = c12p * n1 * n3
        g23 = c12p * n2 * n3

        gamma = np.array(
            [
                [g11, g12, g13],
                [g12, g22, g23],
                [g13, g23, g33],
            ],
            dtype=float,
        )
        return gamma


# ---------------------------- 工具函数 ----------------------------
def _normalize_direction(direction: Iterable[float]) -> np.ndarray:
    """
    归一化传播方向向量。

    Parameters
    ----------
    direction : Iterable[float]
        任意长度可迭代的三个分量。

    Returns
    -------
    ndarray, shape (3,)
        单位向量。
    """
    n = np.asarray(list(direction), dtype=float).reshape(3)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("传播方向向量非法：必须为非零有限向量，形如[1,0,0]")
    return n / norm


def _unit(v: np.ndarray) -> np.ndarray:
    """返回单位向量（数值稳健）。"""
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0])  # 不应出现，兜底
    return v / n
