#!/usr/bin/env python3
import numpy as np
import pytest


@pytest.mark.unit
@pytest.mark.physics
class TestElasticWaveAnalytical:
    """解析弹性波速计算的单元测试。

    覆盖内容：
    - [100]/[110]/[111] 方向的纵横波速
    - [100] 与 [111] 的横波简并性
    - 输出数据结构的健壮性（包含偏振向量）
    """

    def _closed_form_velocities_100(self, C11, C44, rho):
        vL = np.sqrt(C11 / rho)
        vT = np.sqrt(C44 / rho)
        return vL, vT, vT

    def _closed_form_velocities_110(self, C11, C12, C44, rho):
        vL = np.sqrt((C11 + C12 + 2 * C44) / (2 * rho))
        vT2 = np.sqrt(C44 / rho)
        vT1 = np.sqrt((C11 - C12) / (2 * rho))
        # 约定返回顺序：L, T1(较小), T2(较大)
        t1, t2 = sorted([vT1, vT2])
        return vL, t1, t2

    def _closed_form_velocities_111(self, C11, C12, C44, rho):
        vL = np.sqrt((C11 + 2 * C12 + 4 * C44) / (3 * rho))
        vT = np.sqrt((C11 - C12 + C44) / (3 * rho))
        return vL, vT, vT

    @pytest.mark.parametrize(
        "C11,C12,C44,rho,label",
        [
            (110.0, 61.0, 33.0, 2.70, "Al"),
            (175.0, 128.0, 84.0, 8.96, "Cu"),
            (1074.08, 101.73, 641.54, 3.51, "C(diamond)"),
        ],
    )
    def test_velocities_for_standard_directions(self, C11, C12, C44, rho, label):
        from thermoelasticsim.elastic.wave.analytical import ElasticWaveAnalyzer

        ana = ElasticWaveAnalyzer(C11=C11, C12=C12, C44=C44, density=rho)

        # [100]
        res_100 = ana.calculate_wave_velocities([1, 0, 0])
        vL100, vT100a, vT100b = self._closed_form_velocities_100(C11, C44, rho)
        assert pytest.approx(res_100["longitudinal"], rel=1e-6) == vL100
        # 两个横波简并
        assert pytest.approx(res_100["transverse1"], rel=1e-6) == vT100a
        assert pytest.approx(res_100["transverse2"], rel=1e-6) == vT100b

        # [110]
        res_110 = ana.calculate_wave_velocities([1, 1, 0])
        vL110, vT110_1, vT110_2 = self._closed_form_velocities_110(C11, C12, C44, rho)
        assert pytest.approx(res_110["longitudinal"], rel=1e-6) == vL110
        # 返回的两个横波应与闭式解一致（顺序不作强制，按数值排序比对）
        t_calc_sorted = sorted([res_110["transverse1"], res_110["transverse2"]])
        t_true_sorted = sorted([vT110_1, vT110_2])
        assert pytest.approx(t_calc_sorted[0], rel=1e-6) == t_true_sorted[0]
        assert pytest.approx(t_calc_sorted[1], rel=1e-6) == t_true_sorted[1]

        # [111]
        res_111 = ana.calculate_wave_velocities([1, 1, 1])
        vL111, vT111a, vT111b = self._closed_form_velocities_111(C11, C12, C44, rho)
        assert pytest.approx(res_111["longitudinal"], rel=1e-6) == vL111
        # 两个横波简并
        assert pytest.approx(res_111["transverse1"], rel=1e-6) == vT111a
        assert pytest.approx(res_111["transverse2"], rel=1e-6) == vT111b

    def test_polarizations_are_orthonormal(self):
        from thermoelasticsim.elastic.wave.analytical import ElasticWaveAnalyzer

        ana = ElasticWaveAnalyzer(C11=110.0, C12=61.0, C44=33.0, density=2.70)
        res = ana.calculate_wave_velocities([1, 1, 0])
        pol = res["polarizations"]
        eL = np.array(pol["longitudinal"])  # 极化向量
        eT1 = np.array(pol["transverse1"])
        eT2 = np.array(pol["transverse2"])

        # 单位长度
        assert pytest.approx(np.linalg.norm(eL), rel=1e-8, abs=1e-8) == 1.0
        assert pytest.approx(np.linalg.norm(eT1), rel=1e-8, abs=1e-8) == 1.0
        assert pytest.approx(np.linalg.norm(eT2), rel=1e-8, abs=1e-8) == 1.0

        # 两两正交
        assert pytest.approx(float(eL @ eT1), abs=1e-8) == 0.0
        assert pytest.approx(float(eL @ eT2), abs=1e-8) == 0.0
        assert pytest.approx(float(eT1 @ eT2), abs=1e-8) == 0.0

    def test_generate_report_structure(self):
        from thermoelasticsim.elastic.wave.analytical import ElasticWaveAnalyzer

        ana = ElasticWaveAnalyzer(C11=175.0, C12=128.0, C44=84.0, density=8.96)
        report = ana.generate_report()

        # 必须包含三种标准方向
        for k in ("[100]", "[110]", "[111]"):
            assert k in report
            entry = report[k]
            assert set(entry.keys()) >= {
                "longitudinal",
                "transverse1",
                "transverse2",
                "polarizations",
            }
