# tests/test_config.py

"""
@file test_config.py
@brief 测试 config.py 模块中的 ConfigManager 类。
"""

import unittest
import os
from src.python.config import ConfigManager


class TestConfigManager(unittest.TestCase):
    """
    @class TestConfigManager
    @brief 测试 ConfigManager 类。
    """

    def setUp(self) -> None:
        """
        @brief 测试前的初始化。
        """
        self.config_manager = ConfigManager()
        self.test_config_file = "test_config.yaml"
        self.test_config = {
            "unit_cell": {
                "lattice_vectors": [
                    [3.405, 0.0, 0.0],
                    [0.0, 3.405, 0.0],
                    [0.0, 0.0, 3.405],
                ],
                "particles": [
                    {"symbol": "Al", "mass": 26.9815, "position": [0.0, 0.0, 0.0]},
                    {
                        "symbol": "Al",
                        "mass": 26.9815,
                        "position": [1.8075, 1.8075, 1.8075],
                    },
                ],
            },
            "potential": {
                "type": "LennardJones",
                "parameters": {"epsilon": 0.0103, "sigma": 3.405},
                "cutoff": 5.0,
            },
            "optimizer": {"method": "ConjugateGradient"},
            "deformation": {"delta": 0.01},
            "stress_evaluator": {"type": "LennardJones"},
            "md_simulation": {
                "temperature": 300,
                "pressure": 0.0,
                "timestep": 1.0e-3,
                "thermostat": "Nosé-Hoover",
                "barostat": "NoBarostat",
                "steps": 10000,
            },
        }

    def test_save_and_load_config(self) -> None:
        """
        @brief 测试保存和加载配置文件是否正确。
        """
        # 保存配置文件
        self.config_manager.save_config(self.test_config, self.test_config_file)
        self.assertTrue(os.path.exists(self.test_config_file))

        # 加载配置文件
        loaded_config = self.config_manager.load_config(self.test_config_file)
        self.assertEqual(loaded_config, self.test_config)

        # 清理
        os.remove(self.test_config_file)


if __name__ == "__main__":
    unittest.main()
