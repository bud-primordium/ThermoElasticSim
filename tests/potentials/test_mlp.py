#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试机器学习势 (MLP) 占位符
"""
import pytest
from thermoelasticsim.potentials.mlp import MLPotential

def test_mlp_is_placeholder():
    """
    测试 MLPotential 的所有方法是否都按预期引发 NotImplementedError。
    """
    # 尝试初始化，由于 _load_model 未实现，这里就应该失败
    with pytest.raises(NotImplementedError, match="模型加载功能尚未实现"):
        potential = MLPotential(model_path="dummy_path", cutoff=5.0)
        
        # 如果初始化意外成功，则继续测试其他方法
        # （但在当前实现下，代码不会执行到这里）
        with pytest.raises(NotImplementedError):
            potential.calculate_energy(None, None)
        
        with pytest.raises(NotImplementedError):
            potential.calculate_forces(None, None)
