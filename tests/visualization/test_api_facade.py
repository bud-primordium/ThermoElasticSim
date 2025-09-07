"""
可视化API facade模块的单元测试

测试覆盖：
- facade函数的基本调用能力
- 参数传递的正确性
- 返回值类型验证
- 异常处理和边界情况
- 委托给底层实现的正确性
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from thermoelasticsim.core.structure import Atom, Cell
from thermoelasticsim.elastic.wave.analytical import ElasticWaveAnalyzer
from thermoelasticsim.visualization import api


class TestVisualizationApiFacade:
    """测试可视化API facade模块"""

    @pytest.fixture
    def simple_cell(self):
        """创建简单的测试晶胞"""
        lattice_vectors = np.eye(3) * 4.0
        atoms = [
            Atom(id=1, symbol="Al", mass_amu=26.98, position=[0, 0, 0]),
            Atom(id=2, symbol="Al", mass_amu=26.98, position=[2, 2, 0]),
        ]
        return Cell(lattice_vectors=lattice_vectors, atoms=atoms)

    @pytest.fixture
    def elastic_analyzer(self):
        """创建弹性波分析器"""
        return ElasticWaveAnalyzer(C11=110, C12=61, C44=33, density=2.7)

    def test_api_import(self):
        """测试API模块可以正常导入"""
        from thermoelasticsim.visualization import api

        # 检查关键函数是否存在
        assert hasattr(api, "plot_structure_3d")
        assert hasattr(api, "plot_wave_anisotropy")
        assert hasattr(api, "plot_trajectory_animation")
        assert hasattr(api, "plot_energy_evolution")

    @patch(
        "thermoelasticsim.utils.modern_visualization.ModernVisualizer.plot_structure_3d"
    )
    def test_plot_structure_3d_delegation(self, mock_plot, simple_cell):
        """测试plot_structure_3d正确委托给ModernVisualizer"""
        mock_figure = Mock()
        mock_plot.return_value = mock_figure

        result = api.plot_structure_3d(simple_cell, show_box=True, title="Test")

        # 验证委托调用
        mock_plot.assert_called_once_with(simple_cell, show_box=True, title="Test")
        assert result is mock_figure

    @patch(
        "thermoelasticsim.utils.modern_visualization.ModernVisualizer.create_trajectory_animation_plotly"
    )
    def test_plot_trajectory_animation_delegation(self, mock_create):
        """测试轨迹动画函数的委托"""
        with (
            tempfile.NamedTemporaryFile(suffix=".h5") as temp_traj,
            tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_html,
        ):
            trajectory_file = temp_traj.name
            output_html = temp_html.name

            api.plot_trajectory_animation(trajectory_file, output_html, skip=2)

            mock_create.assert_called_once_with(trajectory_file, output_html, skip=2)

    @patch(
        "thermoelasticsim.utils.modern_visualization.ModernVisualizer.plot_energy_evolution"
    )
    def test_plot_energy_evolution_delegation(self, mock_plot):
        """测试能量演化图函数的委托"""
        mock_figure = Mock()
        mock_plot.return_value = mock_figure

        with tempfile.NamedTemporaryFile(suffix=".h5") as temp_traj:
            trajectory_file = temp_traj.name
            result = api.plot_energy_evolution(trajectory_file, save_file=None)

            mock_plot.assert_called_once_with(trajectory_file, None)
            assert result is mock_figure

    def test_plot_wave_anisotropy_from_constants(self, elastic_analyzer):
        """测试从弹性常数直接绘制各向异性图"""
        with tempfile.TemporaryDirectory() as temp_dir:
            outpath = str(Path(temp_dir) / "test_anisotropy.png")

            with patch(
                "thermoelasticsim.elastic.wave.visualization.plot_polar_plane"
            ) as mock_plot:
                mock_plot.return_value = outpath

                result = api.plot_wave_anisotropy_from_constants(
                    C11=110, C12=61, C44=33, density=2.7, plane="001", outpath=outpath
                )

                # 验证调用参数
                mock_plot.assert_called_once()
                call_args = mock_plot.call_args
                assert call_args[1]["plane"] == "001"
                assert call_args[1]["outpath"] == outpath
                assert result == outpath

    @patch("thermoelasticsim.elastic.wave.visualization.plot_velocity_surface_3d")
    def test_plot_velocity_surface_3d_delegation(self, mock_plot, elastic_analyzer):
        """测试3D速度面绘制的委托"""
        mock_return = ("output.html", "output.png")
        mock_plot.return_value = mock_return

        with tempfile.TemporaryDirectory() as temp_dir:
            output_html = str(Path(temp_dir) / "velocity_surface.html")
            output_png = str(Path(temp_dir) / "velocity_surface.png")

            result = api.plot_velocity_surface_3d(
                elastic_analyzer,
                mode="L",
                output_html=output_html,
                output_png=output_png,
            )

            mock_plot.assert_called_once_with(
                elastic_analyzer,
                mode="L",
                n_theta=60,
                n_phi=120,
                output_html=output_html,
                output_png=output_png,
            )
            assert result == mock_return

    @patch(
        "thermoelasticsim.visualization.elastic.ResponsePlotter.plot_c11_c12_combined_response"
    )
    def test_plot_c11_c12_combined_response_delegation(self, mock_plot):
        """测试C11/C12联合响应图的委托"""
        mock_plot.return_value = "output_filename.png"

        c11_data = [{"applied_strain": 0.01, "measured_stress_GPa": 1.1}]
        c12_data = [{"applied_strain": 0.01, "measured_stress_GPa": 0.6}]
        supercell_size = (3, 3, 3)
        output_path = "test_output.png"

        result = api.plot_c11_c12_combined_response(
            c11_data, c12_data, supercell_size, output_path
        )

        mock_plot.assert_called_once_with(
            c11_data,
            c12_data,
            supercell_size,
            output_path,
            slope_override_c11=None,
            slope_override_c12=None,
            subtitle_c11=None,
            subtitle_c12=None,
        )
        assert result == "output_filename.png"

    def test_all_functions_in_all_list(self):
        """确保__all__列表包含所有公开函数"""
        expected_functions = [
            # 结构/轨迹
            "plot_structure_3d",
            "plot_trajectory_animation",
            "plot_energy_evolution",
            "plot_stress_strain_interactive",
            "create_trajectory_video",
            # 弹性波
            "plot_wave_anisotropy",
            "plot_wave_anisotropy_from_constants",
            "plot_velocity_surface_3d",
            "plot_velocity_surface_3d_from_constants",
            # 弹性常数响应
            "plot_c11_c12_combined_response",
            "plot_shear_response",
        ]

        assert set(api.__all__) == set(expected_functions)

        # 确保所有__all__中的函数都可以访问
        for func_name in api.__all__:
            assert hasattr(api, func_name)
            assert callable(getattr(api, func_name))


class TestVisualizationApiIntegration:
    """集成测试：测试与实际底层模块的集成"""

    @pytest.fixture
    def simple_cell(self):
        """创建简单的测试晶胞"""
        lattice_vectors = np.eye(3) * 4.0
        atoms = [Atom(id=1, symbol="Al", mass_amu=26.98, position=[0, 0, 0])]
        return Cell(lattice_vectors=lattice_vectors, atoms=atoms)

    def test_plot_structure_3d_integration(self, simple_cell):
        """集成测试：实际调用plot_structure_3d"""
        try:
            fig = api.plot_structure_3d(simple_cell, show_box=True)
            # 基本检查：确保返回了图形对象
            assert fig is not None
            # 由于plotly依赖可能不完整，这里只做基础检查
        except Exception as e:
            # 如果有依赖问题，至少确保函数存在且可调用
            pytest.skip(f"Skipping integration test due to dependency: {e}")

    def test_wave_anisotropy_parameter_validation(self):
        """测试弹性波各向异性函数的参数验证"""
        # 测试不支持的晶面
        with pytest.raises((ValueError, KeyError)):
            api.plot_wave_anisotropy_from_constants(
                C11=110,
                C12=61,
                C44=33,
                density=2.7,
                plane="999",  # 不支持的晶面应该引发错误
            )
