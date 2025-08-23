"""
基础测试 - 确保包可以正确导入
"""

import pytest


def test_import_thermoelasticsim():
    """测试主包导入"""
    try:
        import thermoelasticsim

        assert hasattr(thermoelasticsim, "__version__")
        assert thermoelasticsim.__version__ == "4.0.0"
    except ImportError as e:
        pytest.fail(f"Failed to import thermoelasticsim: {e}")


def test_import_core():
    """测试核心模块导入"""
    try:
        from thermoelasticsim.core import Atom, Cell, ConfigManager

        # 验证类可以实例化或有正确属性
        assert Atom is not None
        assert Cell is not None
        assert ConfigManager is not None
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")


def test_import_potentials():
    """测试势能模块导入"""
    try:
        from thermoelasticsim.potentials import LennardJonesPotential, Potential

        # 验证类可以实例化或有正确属性
        assert LennardJonesPotential is not None
        assert Potential is not None
    except ImportError as e:
        pytest.fail(f"Failed to import potential modules: {e}")


def test_import_elastic():
    """测试弹性模块导入"""
    try:
        from thermoelasticsim.elastic import ElasticConstantsWorkflow

        # 验证类可以实例化或有正确属性
        assert ElasticConstantsWorkflow is not None
    except ImportError as e:
        pytest.fail(f"Failed to import elastic modules: {e}")
