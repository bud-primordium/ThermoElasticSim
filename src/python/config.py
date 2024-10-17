# src/python/config.py

import yaml


class ConfigManager:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)

    @staticmethod
    def load_config(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return config

    def get(self, key, default=None):
        return self.config.get(key, default)
