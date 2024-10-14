# src/python/config.py

"""
@file config.py
@brief 管理配置参数和全局设置的模块。
"""

import yaml
from typing import Any, Dict


class ConfigManager:
    """
    @class ConfigManager
    @brief 管理项目配置参数的类。
    """

    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        """
        @brief 从配置文件加载配置参数。

        @param config_file 配置文件路径（YAML格式）。
        @return Dict[str, Any]: 配置参数字典。
        """
        with open(config_file, "r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        return config

    @staticmethod
    def save_config(config: Dict[str, Any], config_file: str) -> None:
        """
        @brief 将配置参数保存到配置文件。

        @param config Dict[str, Any]: 配置参数字典。
        @param config_file 配置文件路径（YAML格式）。
        """
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
