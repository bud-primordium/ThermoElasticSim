# tests/test_nose_hoover.py

import pytest
import numpy as np
from python.thermostats import NoseHooverThermostat
from python.structure import Atom, Cell
from python.utils import AMU_TO_EVFSA2


@pytest.fixture
def nose_hoover_thermostat():
    """
    @fixture 定义 Nose-Hoover 恒温器。
    """
    return NoseHooverThermostat(target_temperature=300, time_constant=100)


def test_nose_hoover_apply(nose_hoover_thermostat):
    """
    @brief 测试 Nose-Hoover 恒温器的 apply 方法。
    """
    # 创建一个简单的晶胞，包含两个原子
    atoms = [
        Atom(
            id=0,
            symbol="Al",
            mass=26.9815 * AMU_TO_EVFSA2,  # 转换为 eV/fs^2
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
        ),
        Atom(
            id=1,
            symbol="Al",
            mass=26.9815 * AMU_TO_EVFSA2,  # 转换为 eV/fs^2
            position=np.array([2.55, 0.0, 0.0]),
            velocity=np.array([-1.0, 0.0, 0.0]),
        ),
    ]

    # 初始化力
    for atom in atoms:
        atom.force = np.array([0.0, 0.0, 0.0])

    # 应用恒温器
    dt = 1.0  # fs
    nose_hoover_thermostat.apply(atoms, dt)

    # 检查 xi 是否被更新
    assert nose_hoover_thermostat.xi[0] != 0.0, "xi 未被更新。"

    # 检查速度是否被更新（根据恒温器的实现，速度应被缩放）
    # 由于初始力为零，速度应只受 xi 影响，具体变化取决于 C++ 实现
    # 这里只检查速度是否发生变化
    assert not np.allclose(atoms[0].velocity, [1.0, 0.0, 0.0]), "原子1的速度未被更新。"
    assert not np.allclose(atoms[1].velocity, [-1.0, 0.0, 0.0]), "原子2的速度未被更新。"
