"""
工具模块
"""

__all__ = [
    "TensorConverter", "DataCollector", "NeighborList",
    "AMU_TO_EVFSA2", "EV_TO_GPA", "KB_IN_EV"
]

# 延迟导入避免循环依赖
def __getattr__(name):
    if name == "TensorConverter":
        from .utils import TensorConverter
        return TensorConverter
    elif name == "DataCollector":
        from .utils import DataCollector
        return DataCollector
    elif name == "NeighborList":
        from .utils import NeighborList
        return NeighborList
    elif name == "AMU_TO_EVFSA2":
        from .utils import AMU_TO_EVFSA2
        return AMU_TO_EVFSA2
    elif name == "EV_TO_GPA":
        from .utils import EV_TO_GPA
        return EV_TO_GPA
    elif name == "KB_IN_EV":
        from .utils import KB_IN_EV
        return KB_IN_EV
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
