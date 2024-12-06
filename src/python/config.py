# 文件名: config.py
# 作者: Gilbert Young
# 修改日期: 2024-10-20
# 文件描述: 提供 ConfigManager 类，用于加载和获取配置文件中的参数。

"""
ConfigManager 类

用于加载 YAML 配置文件并获取配置信息
"""

import yaml


class ConfigManager:
    """
    配置管理器，用于加载配置文件并获取参数
    """

    def __init__(self, config_file):
        """加载配置文件"""
        self.config = self.load_config(config_file)

    @staticmethod
    def load_config(config_file):
        """从 YAML 文件加载配置"""
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get(self, key, default=None):
        """获取配置参数，若不存在则返回默认值"""
        return self.config.get(key, default)
