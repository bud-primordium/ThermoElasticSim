#!/usr/bin/env python3
from unittest.mock import patch

import yaml

from thermoelasticsim.cli.run import main


def test_elastic_wave_scenario_dispatch(tmp_path):
    """elastic_wave 场景应调度到对应的pipeline。"""
    config_data = {
        "scenario": "elastic_wave",
        "material": {"symbol": "Al", "structure": "fcc"},
        "wave": {"density": 2.70},
        "run": {"name": "elastic_wave_demo"},
    }
    config_file = tmp_path / "elastic_wave.yaml"
    config_file.write_text(yaml.dump(config_data))

    with patch("thermoelasticsim.cli.run.run_elastic_wave_pipeline") as mock_pipeline:
        result = main(["-c", str(config_file)])
        assert result == 0
        mock_pipeline.assert_called_once()
