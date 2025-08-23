#!/usr/bin/env python3
"""
测试 Tersoff 势占位符
"""

import pytest

from thermoelasticsim.potentials.tersoff import TersoffPotential


@pytest.fixture
def tersoff_potential():
    """提供一个TersoffPotential实例"""
    return TersoffPotential(parameters={}, cutoff=3.0)


def test_tersoff_is_placeholder(tersoff_potential):
    """
    测试 TersoffPotential 的所有方法是否都按预期引发 NotImplementedError。
    """
    with pytest.raises(NotImplementedError, match="Tersoff势的能量计算尚未实现"):
        tersoff_potential.calculate_energy(None, None)

    with pytest.raises(NotImplementedError, match="Tersoff a势的力计算尚未实现"):
        tersoff_potential.calculate_forces(None, None)
