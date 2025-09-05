#!/usr/bin/env python3
import json
import os

import pytest

from thermoelasticsim.cli.pipelines.elastic_wave import run_elastic_wave_pipeline


class _StubCfg:
    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        # 简单的点号键访问
        parts = key.split(".")
        cur = self._data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur


def test_pipeline_outputs_json_and_csv(tmp_path):
    cfg = _StubCfg({"wave": {"density": 2.70}})
    outdir = tmp_path / "out"
    res = run_elastic_wave_pipeline(cfg, str(outdir), material_symbol="Al")

    # 检查返回结构
    assert res["material"] == "Al"
    assert isinstance(res["report"], dict)
    assert set(res["report"].keys()) == {"[100]", "[110]", "[111]"}

    # 检查文件存在且JSON可读
    json_path = res["artifacts"]["json"]
    csv_path = res["artifacts"]["csv"]
    assert os.path.exists(json_path)
    assert os.path.exists(csv_path)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert set(data.keys()) == {"[100]", "[110]", "[111]"}


def test_pipeline_density_from_material(tmp_path):
    # 不提供wave.density，应能从材料参数推导（Al: fcc, a≈4.045 Å）
    cfg = _StubCfg({})
    outdir = tmp_path / "out2"
    res = run_elastic_wave_pipeline(cfg, str(outdir), material_symbol="Al")
    assert res["density"] == pytest.approx(2.70, rel=0.05)


def test_pipeline_generates_polar_plot(tmp_path):
    cfg = _StubCfg(
        {
            "wave": {
                "visualization": {
                    "enabled": True,
                    "plane": "001",
                    "n_angles": 72,
                    "dpi": 100,
                    "output": "analytic_test_polar.png",
                }
            }
        }
    )
    outdir = tmp_path / "out3"
    res = run_elastic_wave_pipeline(cfg, str(outdir), material_symbol="Al")
    polar = res["artifacts"].get("polar")
    assert polar is not None and os.path.exists(polar)


def test_pipeline_handles_surface3d_without_plotly(tmp_path):
    # 即使未安装plotly，也不应异常；仅不生成文件
    cfg = _StubCfg(
        {
            "wave": {
                "visualization": {
                    "enabled": True,
                    "plane": "001",
                    "surface3d": {
                        "enabled": True,
                        "mode": "L",
                        "n_theta": 8,
                        "n_phi": 12,
                    },
                }
            }
        }
    )
    outdir = tmp_path / "out4"
    res = run_elastic_wave_pipeline(cfg, str(outdir), material_symbol="Al")
    # 若环境无plotly，artifacts中可能无surface3d键，不做强校验
    assert isinstance(res, dict)
