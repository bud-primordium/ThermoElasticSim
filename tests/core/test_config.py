#!/usr/bin/env python3
"""配置系统测试模块

测试ConfigManager的配置加载、验证、合并和输出目录管理功能。
"""

import os
from pathlib import Path

import pytest
import yaml

from thermoelasticsim.core.config import ConfigManager


class TestConfigManagerBasic:
    """基本配置加载测试"""

    def test_empty_config_initialization(self):
        """测试空配置初始化"""
        cfg = ConfigManager()
        assert cfg.data == {}

    def test_single_file_loading(self, tmp_path):
        """测试单个YAML文件加载"""
        config_file = tmp_path / "test.yaml"
        config_data = {
            "md": {"temperature": 300.0, "timestep": 0.5},
            "material": {"symbol": "Al", "structure": "fcc"},
        }
        config_file.write_text(yaml.dump(config_data))

        cfg = ConfigManager(files=[str(config_file)])
        assert cfg.get("md.temperature") == 300.0
        assert cfg.get("material.symbol") == "Al"

    def test_multiple_file_merging(self, tmp_path):
        """测试多个配置文件合并"""
        # 基础配置
        base_config = tmp_path / "base.yaml"
        base_data = {
            "md": {"temperature": 300.0, "timestep": 0.5},
            "nvt": {"type": "nhc", "steps": 1000},
        }
        base_config.write_text(yaml.dump(base_data))

        # 覆盖配置
        override_config = tmp_path / "override.yaml"
        override_data = {
            "md": {"temperature": 400.0},  # 覆盖温度
            "nvt": {"steps": 2000},  # 覆盖步数
            "elastic": {"strain_amplitude": 0.01},  # 新增配置
        }
        override_config.write_text(yaml.dump(override_data))

        cfg = ConfigManager(files=[str(base_config), str(override_config)])

        # 验证覆盖
        assert cfg.get("md.temperature") == 400.0  # 被覆盖
        assert cfg.get("md.timestep") == 0.5  # 保持原值
        assert cfg.get("nvt.steps") == 2000  # 被覆盖
        assert cfg.get("nvt.type") == "nhc"  # 保持原值
        assert cfg.get("elastic.strain_amplitude") == 0.01  # 新增

    def test_nonexistent_file_handling(self):
        """测试不存在文件的处理"""
        # ConfigManager实际上会跳过不存在的文件，而不是抛出异常
        cfg = ConfigManager(files=["nonexistent.yaml"])
        assert cfg.data == {}  # 应该得到空配置


class TestConfigManagerAccess:
    """配置访问和查询测试"""

    @pytest.fixture
    def sample_config(self, tmp_path):
        """创建示例配置"""
        config_file = tmp_path / "sample.yaml"
        config_data = {
            "scenario": "nvt",
            "md": {"temperature": 300.0, "timestep": 0.5},
            "nvt": {
                "type": "langevin",
                "friction": 1.0,
                "nested": {"deep": {"value": 42}},
            },
            "list_param": [1, 2, 3, 4, 5],
        }
        config_file.write_text(yaml.dump(config_data))
        return ConfigManager(files=[str(config_file)])

    def test_simple_path_access(self, sample_config):
        """测试简单路径访问"""
        assert sample_config.get("scenario") == "nvt"
        assert sample_config.get("md.temperature") == 300.0
        assert sample_config.get("nvt.friction") == 1.0

    def test_nested_path_access(self, sample_config):
        """测试深层嵌套路径访问"""
        assert sample_config.get("nvt.nested.deep.value") == 42

    def test_default_value_handling(self, sample_config):
        """测试默认值处理"""
        assert sample_config.get("nonexistent.path") is None
        assert sample_config.get("nonexistent.path", "default") == "default"
        assert sample_config.get("md.nonexistent", 100.0) == 100.0

    def test_list_access(self, sample_config):
        """测试列表类型访问"""
        assert sample_config.get("list_param") == [1, 2, 3, 4, 5]

    def test_type_preservation(self, sample_config):
        """测试数据类型保持"""
        assert isinstance(sample_config.get("md.temperature"), float)
        assert isinstance(sample_config.get("nvt.nested.deep.value"), int)
        assert isinstance(sample_config.get("list_param"), list)


class TestConfigManagerOutput:
    """输出目录管理测试"""

    def test_output_directory_creation(self, tmp_path):
        """测试输出目录创建"""
        cfg = ConfigManager()

        # 临时设置输出基础路径
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            output_dir = cfg.make_output_dir("test_run")

            assert Path(output_dir).exists()
            assert Path(output_dir).is_dir()
            assert "test_run" in output_dir

        finally:
            os.chdir(original_cwd)

    def test_config_snapshot_saving(self, tmp_path):
        """测试配置快照保存"""
        config_file = tmp_path / "test.yaml"
        config_data = {"test": {"value": 123}}
        config_file.write_text(yaml.dump(config_data))

        cfg = ConfigManager(files=[str(config_file)])

        # 临时设置工作目录
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            output_dir = cfg.make_output_dir("snapshot_test")

            # 检查配置文件是否被复制（实际文件名是resolved_config.yaml）
            snapshot_path = Path(output_dir) / "resolved_config.yaml"

            # 配置快照需要手动调用
            cfg.snapshot(output_dir)
            assert snapshot_path.exists()

            # 验证快照内容
            with open(snapshot_path) as f:
                snapshot_data = yaml.safe_load(f)
            assert snapshot_data["test"]["value"] == 123

        finally:
            os.chdir(original_cwd)


class TestConfigManagerSeed:
    """随机种子设置测试"""

    def test_default_seed_setting(self):
        """测试默认种子设置"""
        cfg = ConfigManager()
        seed = cfg.set_global_seed()
        assert isinstance(seed, int)
        assert 0 <= seed <= 2**32 - 1

    def test_explicit_seed_setting(self, tmp_path):
        """测试显式种子设置"""
        config_file = tmp_path / "seed.yaml"
        config_data = {"rng": {"global_seed": 12345}}
        config_file.write_text(yaml.dump(config_data))

        cfg = ConfigManager(files=[str(config_file)])
        seed = cfg.set_global_seed()
        assert seed == 12345

    def test_seed_reproducibility(self, tmp_path):
        """测试种子可重现性"""
        config_file = tmp_path / "repro.yaml"
        config_data = {"rng": {"global_seed": 42}}
        config_file.write_text(yaml.dump(config_data))

        # 两次加载相同配置应该得到相同种子
        cfg1 = ConfigManager(files=[str(config_file)])
        seed1 = cfg1.set_global_seed()

        cfg2 = ConfigManager(files=[str(config_file)])
        seed2 = cfg2.set_global_seed()

        assert seed1 == seed2 == 42


class TestConfigManagerEdgeCases:
    """边界情况和错误处理测试"""

    def test_empty_yaml_file(self, tmp_path):
        """测试空YAML文件"""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        cfg = ConfigManager(files=[str(empty_file)])
        assert cfg.data == {}

    def test_malformed_yaml_handling(self, tmp_path):
        """测试格式错误的YAML文件"""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("invalid: yaml: content: [unclosed")

        with pytest.raises(yaml.YAMLError):
            ConfigManager(files=[str(bad_file)])

    def test_path_with_empty_segments(self, tmp_path):
        """测试包含空段的路径"""
        config_file = tmp_path / "test.yaml"
        config_data = {"test": {"value": 123}}
        config_file.write_text(yaml.dump(config_data))

        cfg = ConfigManager(files=[str(config_file)])

        # 路径中的空段应该被正确处理
        assert cfg.get("test..value") is None  # 双点
        assert cfg.get(".test.value") is None  # 开头的点
        assert cfg.get("test.value.") is None  # 结尾的点
