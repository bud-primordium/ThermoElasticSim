# tests/test_utils.py

import pytest
import numpy as np
from python.utils import TensorConverter, DataCollector
from python.structure import Atom, Cell


def test_tensor_converter_to_voigt():
    """
    @brief 测试 TensorConverter 的 to_voigt 方法
    """
    tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    expected_voigt = np.array([1.0, 5.0, 9.0, 6.0, 3.0, 2.0])
    voigt = TensorConverter.to_voigt(tensor)
    np.testing.assert_array_almost_equal(voigt, expected_voigt, decimal=6)


@pytest.fixture
def cell_fixture():
    """
    @fixture 创建一个简单的晶胞，用于数据收集测试
    """
    lattice_vectors = np.eye(3) * 4.05  # Å
    mass_amu = 26.9815  # amu
    position = np.array([0.0, 0.0, 0.0])
    atom = Atom(id=0, symbol="Al", mass_amu=mass_amu, position=position)
    cell = Cell(lattice_vectors, [atom], pbc_enabled=True)
    return cell


def test_data_collector(cell_fixture):
    """
    @brief 测试 DataCollector 的数据收集功能
    """
    collector = DataCollector()
    collector.collect(cell_fixture)
    assert len(collector.data) == 1
    collected = collector.data[0]
    assert "positions" in collected
    assert "velocities" in collected
    assert len(collected["positions"]) == 1
    assert len(collected["velocities"]) == 1
    np.testing.assert_array_equal(
        collected["positions"][0], cell_fixture.atoms[0].position
    )
    np.testing.assert_array_equal(
        collected["velocities"][0], cell_fixture.atoms[0].velocity
    )
