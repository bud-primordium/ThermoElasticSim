"""
测试势函数基类
"""

import pytest
from thermoelasticsim.potentials.base import Potential


def test_potential_is_abstract():
    """
    测试 Potential 基类不能被直接实例化。
    """
    # 抽象基类不应能被实例化，否则会抛出 TypeError
    with pytest.raises(TypeError, match="Can't instantiate abstract class Potential"):
        Potential(parameters={}, cutoff=5.0)
