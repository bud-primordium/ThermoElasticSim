#!/usr/bin/env python3
"""CLI run模块测试

测试YAML配置驱动的CLI入口功能，包括场景调度和参数解析。
"""

from unittest.mock import patch

import pytest
import yaml

from thermoelasticsim.cli.run import main


class TestCLIRunBasic:
    """CLI基本功能测试"""

    def test_missing_config_argument(self):
        """测试缺少配置文件参数时的错误处理"""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2  # argparse错误码

    def test_nonexistent_config_file(self, tmp_path):
        """测试不存在的配置文件处理"""
        # CLI实际上会使用默认配置，不会抛出异常
        # Mock pipeline避免实际执行
        with patch("thermoelasticsim.cli.run.run_zero_temp_pipeline") as mock_pipeline:
            result = main(["-c", "nonexistent.yaml"])
            assert result == 0
            mock_pipeline.assert_called_once()

    def test_config_argument_parsing(self, tmp_path):
        """测试配置文件参数解析"""
        config_file = tmp_path / "test.yaml"
        config_data = {
            "scenario": "nve",
            "material": {"symbol": "Al", "structure": "fcc"},
            "run": {"name": "test_run"},
        }
        config_file.write_text(yaml.dump(config_data))

        # Mock the pipeline function to avoid actual execution
        with patch("thermoelasticsim.cli.run.run_nve_pipeline") as mock_pipeline:
            result = main(["-c", str(config_file)])
            assert result == 0  # 成功退出码
            mock_pipeline.assert_called_once()


class TestCLIScenarioDispatch:
    """场景调度测试"""

    @pytest.fixture
    def base_config_data(self):
        """基础配置数据"""
        return {
            "material": {"symbol": "Al", "structure": "fcc"},
            "potential": "EAM_Al1",
            "supercell": [2, 2, 2],
            "run": {"name": "test_scenario"},
        }

    def test_relax_scenario_dispatch(self, tmp_path, base_config_data):
        """测试relax场景调度"""
        config_data = {**base_config_data, "scenario": "relax"}
        config_file = tmp_path / "relax.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("thermoelasticsim.cli.run.run_relax_pipeline") as mock_pipeline:
            result = main(["-c", str(config_file)])
            assert result == 0
            mock_pipeline.assert_called_once()

    def test_nve_scenario_dispatch(self, tmp_path, base_config_data):
        """测试NVE场景调度"""
        config_data = {**base_config_data, "scenario": "nve"}
        config_file = tmp_path / "nve.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("thermoelasticsim.cli.run.run_nve_pipeline") as mock_pipeline:
            result = main(["-c", str(config_file)])
            assert result == 0
            mock_pipeline.assert_called_once()

    def test_nvt_scenario_dispatch(self, tmp_path, base_config_data):
        """测试NVT场景调度"""
        config_data = {**base_config_data, "scenario": "nvt"}
        config_file = tmp_path / "nvt.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("thermoelasticsim.cli.run.run_nvt_pipeline") as mock_pipeline:
            result = main(["-c", str(config_file)])
            assert result == 0
            mock_pipeline.assert_called_once()

    def test_npt_scenario_dispatch(self, tmp_path, base_config_data):
        """测试NPT场景调度"""
        config_data = {**base_config_data, "scenario": "npt"}
        config_file = tmp_path / "npt.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("thermoelasticsim.cli.run.run_npt_pipeline") as mock_pipeline:
            result = main(["-c", str(config_file)])
            assert result == 0
            mock_pipeline.assert_called_once()

    def test_zero_temp_scenario_dispatch(self, tmp_path, base_config_data):
        """测试零温弹性场景调度"""
        config_data = {**base_config_data, "scenario": "zero_temp"}
        config_file = tmp_path / "zero_temp.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch("thermoelasticsim.cli.run.run_zero_temp_pipeline") as mock_pipeline:
            result = main(["-c", str(config_file)])
            assert result == 0
            mock_pipeline.assert_called_once()

    def test_finite_temp_scenario_dispatch(self, tmp_path, base_config_data):
        """测试有限温弹性场景调度"""
        config_data = {**base_config_data, "scenario": "finite_temp"}
        config_file = tmp_path / "finite_temp.yaml"
        config_file.write_text(yaml.dump(config_data))

        with patch(
            "thermoelasticsim.cli.run.run_finite_temp_pipeline"
        ) as mock_pipeline:
            result = main(["-c", str(config_file)])
            assert result == 0
            mock_pipeline.assert_called_once()

    def test_unknown_scenario_handling(self, tmp_path, base_config_data):
        """测试未知场景处理"""
        config_data = {**base_config_data, "scenario": "unknown_scenario"}
        config_file = tmp_path / "unknown.yaml"
        config_file.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="未知场景类型"):
            main(["-c", str(config_file)])


class TestCLIConfiguration:
    """CLI配置处理测试"""

    def test_seed_setting_integration(self, tmp_path):
        """测试种子设置集成"""
        config_data = {
            "scenario": "nve",
            "material": {"symbol": "Al", "structure": "fcc"},
            "rng": {"global_seed": 42},
            "run": {"name": "seed_test"},
        }
        config_file = tmp_path / "seed.yaml"
        config_file.write_text(yaml.dump(config_data))

        with (
            patch("thermoelasticsim.cli.run.run_nve_pipeline") as mock_pipeline,
            patch(
                "thermoelasticsim.core.config.ConfigManager.set_global_seed"
            ) as mock_seed,
        ):
            mock_seed.return_value = 42

            result = main(["-c", str(config_file)])
            assert result == 0
            mock_seed.assert_called_once()
            mock_pipeline.assert_called_once()

    def test_output_directory_creation(self, tmp_path):
        """测试输出目录创建"""
        config_data = {
            "scenario": "nve",
            "material": {"symbol": "Al", "structure": "fcc"},
            "run": {"name": "output_test"},
        }
        config_file = tmp_path / "output.yaml"
        config_file.write_text(yaml.dump(config_data))

        with (
            patch("thermoelasticsim.cli.run.run_nve_pipeline") as mock_pipeline,
            patch(
                "thermoelasticsim.core.config.ConfigManager.make_output_dir"
            ) as mock_outdir,
        ):
            mock_outdir.return_value = str(tmp_path / "output")

            result = main(["-c", str(config_file)])
            assert result == 0
            mock_outdir.assert_called_once_with("output_test")
            mock_pipeline.assert_called_once()


class TestCLIEdgeCases:
    """边界情况测试"""

    def test_malformed_yaml_handling(self, tmp_path):
        """测试格式错误YAML文件的处理"""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            main(["-c", str(bad_config)])

    def test_empty_config_handling(self, tmp_path):
        """测试空配置文件处理"""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        # 空配置使用默认scenario为zero_temp，不会抛出异常
        with patch("thermoelasticsim.cli.run.run_zero_temp_pipeline") as mock_pipeline:
            result = main(["-c", str(empty_config)])
            assert result == 0
            mock_pipeline.assert_called_once()

    def test_missing_scenario_field(self, tmp_path):
        """测试缺少scenario字段的处理"""
        config_data = {
            "material": {"symbol": "Al", "structure": "fcc"},
            "run": {"name": "no_scenario"},
            # 缺少scenario字段
        }
        config_file = tmp_path / "no_scenario.yaml"
        config_file.write_text(yaml.dump(config_data))

        # 缺少scenario字段使用默认值，不会抛出异常
        with patch("thermoelasticsim.cli.run.run_zero_temp_pipeline") as mock_pipeline:
            result = main(["-c", str(config_file)])
            assert result == 0
            mock_pipeline.assert_called_once()
