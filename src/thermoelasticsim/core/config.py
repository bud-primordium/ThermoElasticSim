# 文件名: config.py
# 作者: Gilbert Young
# 修改日期: 2025-03-27
# 文件描述: 提供 ConfigManager 类，用于加载和获取配置文件中的参数。

"""
配置模块

包含 ConfigManager 类，用于加载和管理 YAML 格式的配置文件

Classes:
    ConfigManager: 加载和访问配置文件的配置管理器
"""

import yaml
from typing import Any, Optional, Dict


class ConfigManager:
    """加载和管理 YAML 配置文件的配置管理器

    Attributes
    ----------
    config : dict
        存储加载的配置数据
    """

    def __init__(self, config_file: str) -> None:
        """初始化配置管理器并加载配置文件

        Parameters
        ----------
        config_file : str
            配置文件的路径

        Raises
        ------
        FileNotFoundError
            如果配置文件不存在
        yaml.YAMLError
            如果配置文件不是有效的YAML格式
        """
        self.config = self.load_config(config_file)

    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        """从 YAML 文件加载配置数据

        Parameters
        ----------
        config_file : str
            配置文件的路径

        Returns
        -------
        dict
            包含配置数据的字典

        Raises
        ------
        FileNotFoundError
            如果配置文件不存在
        yaml.YAMLError
            如果配置文件不是有效的YAML格式

        Notes
        -----
        使用yaml.safe_load()来避免潜在的安全问题
        """
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """获取配置参数的值

        Parameters
        ----------
        key : str
            要获取的配置键名
        default : Any, optional
            如果键不存在时返回的默认值 (默认: None)

        Returns
        -------
        Any
            配置值或默认值
        """
        return self.config.get(key, default)
