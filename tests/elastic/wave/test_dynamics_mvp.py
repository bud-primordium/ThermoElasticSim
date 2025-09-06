#!/usr/bin/env python3
"""Phase B MVP - 动力学基础单元测试

避免重型MD，聚焦于：
- 平面波激发器的基本性质（模式、去质心）
- x 方向分箱的形状与有序性
"""

import numpy as np

from thermoelasticsim.core import CrystallineStructureBuilder
from thermoelasticsim.elastic.wave.dynamics import (
    WaveExcitation,
    _bin_average_signal,
)


def _make_small_cell():
    builder = CrystallineStructureBuilder()
    # 小超胞，确保运行迅速
    return builder.create_fcc("Al", 4.045, (4, 4, 4))


def test_plane_wave_excitation_zero_com_velocity():
    cell = _make_small_cell()
    exc = WaveExcitation(
        direction="x", polarization="L", n_waves=2, amplitude_velocity=1e-3
    )
    exc.apply(cell)

    # 质心速度应近似为零
    vs = np.array([a.velocity for a in cell.atoms])
    ms = np.array([a.mass for a in cell.atoms])
    v_com = (ms[:, None] * vs).sum(axis=0) / ms.sum()
    assert np.linalg.norm(v_com) < 1e-10


def test_bin_average_signal_shapes_and_order():
    cell = _make_small_cell()
    # 构造一个沿 x 的线性场作为“位移”信号
    R = cell.get_positions()
    u = R[:, 0].copy()
    x_centers, avg = _bin_average_signal(cell, u, n_bins=16)
    assert x_centers.shape == (16,)
    assert avg.shape == (16,)
    assert np.all(np.diff(x_centers) > 0.0)
