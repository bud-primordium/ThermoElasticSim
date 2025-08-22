"""
核心模块 - 基础数据结构和配置管理
"""

__all__ = ["Atom", "Cell", "ConfigManager"]

# 延迟导入避免循环依赖
def __getattr__(name):
    if name == "Atom":
        from .structure import Atom
        return Atom
    elif name == "Cell":
        from .structure import Cell
        return Cell
    elif name == "ConfigManager":
        from .config import ConfigManager
        return ConfigManager
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
