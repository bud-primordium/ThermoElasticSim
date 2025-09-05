#!/usr/bin/env python3
import os

import numpy as np

from thermoelasticsim.elastic.wave.analytical import ElasticWaveAnalyzer
from thermoelasticsim.elastic.wave.visualization import (
    compute_velocities_over_directions,
    plot_polar_plane,
    sample_plane_directions,
)


def _make_ana():
    # Al 参数，ρ≈2.70 g/cm^3
    return ElasticWaveAnalyzer(C11=110.0, C12=61.0, C44=33.0, density=2.70)


def test_sample_plane_directions_shapes():
    theta, dirs = sample_plane_directions("001", 180)
    assert theta.shape == (180,)
    assert dirs.shape == (180, 3)
    # 单位化
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_compute_velocities_over_directions_shapes():
    ana = _make_ana()
    theta, dirs = sample_plane_directions("110", 72)
    v = compute_velocities_over_directions(ana, dirs)
    assert set(v.keys()) == {"vL", "vT1", "vT2"}
    assert v["vL"].shape == (72,)
    assert v["vT1"].shape == (72,)
    assert v["vT2"].shape == (72,)


def test_plot_polar_plane_output(tmp_path):
    ana = _make_ana()
    out = tmp_path / "polar.png"
    path = plot_polar_plane(ana, plane="111", n_angles=90, outpath=str(out), dpi=120)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
