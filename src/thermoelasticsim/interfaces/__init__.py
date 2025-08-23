"""C++接口模块"""

__all__ = ["CppInterface"]


# 延迟导入避免循环依赖
def __getattr__(name):
    if name == "CppInterface":
        from .cpp_interface import CppInterface

        return CppInterface
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
