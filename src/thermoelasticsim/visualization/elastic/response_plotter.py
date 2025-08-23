#!/usr/bin/env python3
"""
弹性常数响应曲线绘制器

从v7 system_size_comparison_v7.py提取的绘图功能，重构为模块化、可复用的组件。
支持C11、C12、C44等所有弹性常数的应力-应变响应可视化。

Author: Gilbert Young
Created: 2025-08-15
"""

from pathlib import Path

import numpy as np

from thermoelasticsim.utils.plot_config import plt


class ResponsePlotter:
    """
    弹性常数响应曲线绘制器

    提供统一的绘图接口，支持：
    - C11/C12联合响应图
    - C44/C55/C66剪切响应图
    - 弹性常数对比图
    - 收敛性分析图

    Examples
    --------
    >>> plotter = ResponsePlotter()
    >>> plotter.plot_c11_c12_response(c11_data, c12_data, 'output.png')
    >>> plotter.plot_shear_response(c44_data, 'shear_output.png')
    """

    def __init__(self, dpi: int = 300, figsize_scale: float = 1.0):
        """
        初始化绘图器

        Parameters
        ----------
        dpi : int
            图像分辨率
        figsize_scale : float
            图像尺寸缩放因子
        """
        self.dpi = dpi
        self.figsize_scale = figsize_scale

        # 预定义颜色和标记
        self.colors = {
            "C11": "#2E86C1",
            "C12": "#E74C3C",
            "C44": "#2E86C1",
            "C55": "#E74C3C",
            "C66": "#58D68D",
        }

        self.markers = {"C11": "o", "C12": "s", "C44": "o", "C55": "s", "C66": "^"}

        # 文献值 (GPa)
        self.literature_values = {
            "C11": 110.0,
            "C12": 61.0,
            "C44": 33.0,
            "C55": 33.0,
            "C66": 33.0,
        }

    def plot_c11_c12_combined_response(
        self,
        c11_data: list[dict],
        c12_data: list[dict],
        supercell_size: tuple[int, int, int],
        output_path: str,
    ) -> str:
        """
        生成C11/C12联合应力-应变响应关系图

        从v7的plot_c11_c12_combined_response提取和重构

        Parameters
        ----------
        c11_data : List[Dict]
            C11数据点列表
        c12_data : List[Dict]
            C12数据点列表
        supercell_size : Tuple[int, int, int]
            系统尺寸
        output_path : str
            输出文件路径

        Returns
        -------
        str
            生成的图像文件名
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(16 * self.figsize_scale, 12 * self.figsize_scale)
        )

        # 准备C11数据
        c11_strains = [row["applied_strain"] for row in c11_data]
        c11_stresses = [row["measured_stress_GPa"] for row in c11_data]
        c11_converged_states = [row["optimization_converged"] for row in c11_data]

        # 准备C12数据
        c12_strains = [row["applied_strain"] for row in c12_data]
        c12_stresses = [row["measured_stress_GPa"] for row in c12_data]
        c12_converged_states = [row["optimization_converged"] for row in c12_data]

        # C11图
        self._plot_stress_strain_scatter(
            ax1,
            c11_strains,
            c11_stresses,
            c11_converged_states,
            "C11",
            self.literature_values["C11"],
            "εxx",
            "σxx",
            "C11: xx应变→xx应力",
        )

        # C12图
        self._plot_stress_strain_scatter(
            ax2,
            c12_strains,
            c12_stresses,
            c12_converged_states,
            "C12",
            self.literature_values["C12"],
            "εxx",
            "σyy",
            "C12: xx应变→yy应力",
        )

        # C11/C12对比图
        self._plot_elastic_constant_bar(
            ax3,
            c11_strains,
            c11_stresses,
            c11_converged_states,
            "C11",
            self.literature_values["C11"],
        )

        self._plot_elastic_constant_bar(
            ax4,
            c12_strains,
            c12_stresses,
            c12_converged_states,
            "C12",
            self.literature_values["C12"],
        )

        plt.suptitle(
            f"C11/C12联合计算 - {supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}系统",
            fontsize=16 * self.figsize_scale,
            weight="bold",
        )
        plt.tight_layout()

        # 保存图片
        filepath = Path(output_path)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return filepath.name

    def plot_shear_response(
        self,
        detailed_results: list[dict],
        supercell_size: tuple[int, int, int],
        output_path: str,
    ) -> str:
        """
        生成C44/C55/C66剪切应力-应变响应关系图

        从v7的plot_stress_strain_response提取和重构

        Parameters
        ----------
        detailed_results : List[Dict]
            详细的剪切响应结果
        supercell_size : Tuple[int, int, int]
            系统尺寸
        output_path : str
            输出文件路径

        Returns
        -------
        str
            生成的图像文件名
        """
        fig = plt.figure(figsize=(18 * self.figsize_scale, 12 * self.figsize_scale))

        # 创建2行2列子图布局
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3, height_ratios=[3, 2])
        ax_shear = [fig.add_subplot(gs[0, i]) for i in range(3)]  # 上排3个剪切子图
        ax_summary = fig.add_subplot(gs[1, :])  # 下排汇总图

        # 准备数据
        directions = ["yz(C44)", "xz(C55)", "xy(C66)"]
        colors = [self.colors["C44"], self.colors["C55"], self.colors["C66"]]
        markers = [self.markers["C44"], self.markers["C55"], self.markers["C66"]]
        literature_values = [
            self.literature_values["C44"],
            self.literature_values["C55"],
            self.literature_values["C66"],
        ]

        # 为每个剪切模式单独绘制子图
        elastic_constants = []
        convergence_quality = []

        for i, result in enumerate(detailed_results):
            ax = ax_shear[i]
            direction = result["direction"]
            color = colors[i]
            marker = markers[i]
            lit_value = literature_values[i]

            # 获取多点数据
            strains = result["strains"]
            stresses = result["stresses"]
            converged_states = result["converged_states"]

            # 绘制散点图和拟合线
            elastic_constant = self._plot_shear_scatter(
                ax,
                strains,
                stresses,
                converged_states,
                direction,
                color,
                marker,
                lit_value,
            )

            elastic_constants.append(elastic_constant)
            convergence_quality.append(sum(converged_states) / len(converged_states))

        # 下方汇总图：弹性常数对比
        self._plot_elastic_constants_summary(
            ax_summary,
            directions,
            elastic_constants,
            convergence_quality,
            colors,
            literature_values[0],
            supercell_size,
        )

        plt.tight_layout()

        # 保存图片
        filepath = Path(output_path)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return filepath.name

    def _plot_stress_strain_scatter(
        self,
        ax,
        strains: list[float],
        stresses: list[float],
        converged_states: list[bool],
        constant_name: str,
        literature_value: float,
        strain_label: str,
        stress_label: str,
        title: str,
    ) -> float:
        """绘制应力-应变散点图和拟合线"""
        # 分别绘制收敛和不收敛的点
        converged_strains = [
            s for s, c in zip(strains, converged_states, strict=False) if c
        ]
        converged_stresses = [
            st for st, c in zip(stresses, converged_states, strict=False) if c
        ]
        failed_strains = [
            s for s, c in zip(strains, converged_states, strict=False) if not c
        ]
        failed_stresses = [
            st for st, c in zip(stresses, converged_states, strict=False) if not c
        ]

        color = self.colors[constant_name]
        marker = self.markers[constant_name]

        # 收敛点：实心符号
        if converged_strains:
            ax.scatter(
                converged_strains,
                converged_stresses,
                marker=marker,
                color=color,
                s=80,
                label=f"{constant_name} (收敛)",
                alpha=0.8,
                edgecolors="black",
            )

        # 不收敛点：空心符号
        if failed_strains:
            ax.scatter(
                failed_strains,
                failed_stresses,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                s=80,
                label=f"{constant_name} (未收敛)",
                alpha=0.8,
                linewidth=2,
            )

        # 添加文献值理论斜率参考线
        strain_range = np.linspace(-0.003, 0.003, 100)
        theory_stress = literature_value * strain_range
        ax.plot(
            strain_range,
            theory_stress,
            "k:",
            linewidth=2,
            alpha=0.7,
            label=f"理论斜率 ({literature_value} GPa)",
        )

        # 线性拟合（只用收敛点）
        fitted_constant = 0.0
        if len(converged_strains) >= 2:
            coeffs = np.polyfit(converged_strains, converged_stresses, 1)
            fit_strains = np.linspace(
                min(converged_strains), max(converged_strains), 100
            )
            fit_stresses = np.polyval(coeffs, fit_strains)
            ax.plot(
                fit_strains,
                fit_stresses,
                "--",
                color=color,
                alpha=0.7,
                linewidth=2,
                label=f"拟合斜率 ({coeffs[0]:.1f} GPa)",
            )
            fitted_constant = coeffs[0]

            # 计算R²
            y_pred = np.polyval(coeffs, converged_strains)
            ss_res = np.sum((converged_stresses - y_pred) ** 2)
            ss_tot = np.sum((converged_stresses - np.mean(converged_stresses)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            # 在图上显示拟合质量
            ax.text(
                0.05,
                0.95,
                f"R² = {r_squared:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        ax.set_xlabel(strain_label, fontsize=12)
        ax.set_ylabel(f"{stress_label} (GPa)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")

        return fitted_constant

    def _plot_shear_scatter(
        self,
        ax,
        strains: list[float],
        stresses: list[float],
        converged_states: list[bool],
        direction: str,
        color: str,
        marker: str,
        lit_value: float,
    ) -> float:
        """绘制剪切模式的散点图"""
        # 分别绘制收敛和不收敛的点
        converged_strains = [
            s for s, c in zip(strains, converged_states, strict=False) if c
        ]
        converged_stresses = [
            st for st, c in zip(stresses, converged_states, strict=False) if c
        ]
        failed_strains = [
            s for s, c in zip(strains, converged_states, strict=False) if not c
        ]
        failed_stresses = [
            st for st, c in zip(stresses, converged_states, strict=False) if not c
        ]

        # 收敛点：实心符号
        if converged_strains:
            ax.scatter(
                converged_strains,
                converged_stresses,
                marker=marker,
                color=color,
                s=100,
                label="收敛点",
                alpha=0.8,
                edgecolors="black",
                linewidth=1,
            )

        # 不收敛点：空心符号
        if failed_strains:
            ax.scatter(
                failed_strains,
                failed_stresses,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                s=100,
                label="未收敛",
                alpha=0.8,
                linewidth=2,
            )

        # 添加文献值理论斜率参考线
        if strains:
            strain_range = np.linspace(min(strains) * 1.2, max(strains) * 1.2, 100)
            theory_stress = lit_value * strain_range
            ax.plot(
                strain_range,
                theory_stress,
                "k:",
                linewidth=2,
                alpha=0.7,
                label=f"理论斜率 ({lit_value} GPa)",
            )

        # 只对收敛点进行线性拟合
        elastic_constant = 0.0
        if len(converged_strains) >= 2:
            coeffs = np.polyfit(converged_strains, converged_stresses, 1)
            fit_strains = np.linspace(
                min(converged_strains), max(converged_strains), 100
            )
            fit_stresses = np.polyval(coeffs, fit_strains)
            ax.plot(
                fit_strains,
                fit_stresses,
                "--",
                color=color,
                alpha=0.9,
                linewidth=3,
                label=f"拟合 ({coeffs[0]:.1f} GPa)",
            )
            elastic_constant = coeffs[0]

            # 计算R²
            y_pred = np.polyval(coeffs, converged_strains)
            ss_res = np.sum((converged_stresses - y_pred) ** 2)
            ss_tot = np.sum((converged_stresses - np.mean(converged_stresses)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

            # 在图上显示拟合质量
            ax.text(
                0.05,
                0.95,
                f"R² = {r_squared:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        # 设置子图属性
        shear_component = direction.split("(")[0]  # yz, xz, xy
        ax.set_xlabel(f"{shear_component}剪切应变", fontsize=12)
        ax.set_ylabel(f"{shear_component}剪切应力 (GPa)", fontsize=12)
        ax.set_title(f"{direction}", fontsize=14, weight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")

        return elastic_constant

    def _plot_elastic_constant_bar(
        self,
        ax,
        strains: list[float],
        stresses: list[float],
        converged_states: list[bool],
        constant_name: str,
        literature_value: float,
    ) -> float:
        """绘制单个弹性常数对比柱状图"""
        # 计算拟合值
        converged_strains = [
            s for s, c in zip(strains, converged_states, strict=False) if c
        ]
        converged_stresses = [
            st for st, c in zip(stresses, converged_states, strict=False) if c
        ]

        fitted_value = 0.0
        convergence_rate = sum(converged_states) / len(converged_states)

        if len(converged_strains) >= 2:
            coeffs = np.polyfit(converged_strains, converged_stresses, 1)
            fitted_value = coeffs[0]

        color = self.colors[constant_name]

        bar = ax.bar(
            [constant_name],
            [fitted_value],
            color=color,
            alpha=0.3 + 0.7 * convergence_rate,
            edgecolor="black",
            linewidth=1,
            width=0.6,
        )

        # 文献值参考线
        ax.axhline(
            y=literature_value,
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"文献值 ({literature_value} GPa)",
        )

        # 数值标签
        if fitted_value > 0:
            height = bar[0].get_height()
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2.0,
                height + max(height * 0.02, 2),
                f"{fitted_value:.1f}",
                ha="center",
                va="bottom",
                fontsize=14,
                weight="bold",
            )

            # 误差
            error = (fitted_value - literature_value) / literature_value * 100
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2.0,
                height + max(height * 0.08, 8),
                f"({error:+.1f}%)",
                ha="center",
                va="bottom",
                fontsize=12,
                color="gray",
            )

            # 收敛率
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2.0,
                height / 2,
                f"{convergence_rate:.0%}",
                ha="center",
                va="center",
                fontsize=12,
                color="white",
                weight="bold",
            )

        ax.set_ylabel("弹性常数 (GPa)", fontsize=12)
        ax.set_title(f"{constant_name}计算结果", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=10)
        ax.set_ylim(
            0, max(fitted_value if fitted_value > 0 else 0, literature_value) * 1.3
        )

        return fitted_value

    def _plot_elastic_constants_summary(
        self,
        ax,
        directions: list[str],
        elastic_constants: list[float],
        convergence_quality: list[float],
        colors: list[str],
        literature_value: float,
        supercell_size: tuple[int, int, int],
    ):
        """绘制弹性常数汇总对比图"""
        x_pos = np.arange(len(directions))
        bars = ax.bar(
            x_pos,
            elastic_constants,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # 根据收敛质量调整透明度
        for bar, quality in zip(bars, convergence_quality, strict=False):
            bar.set_alpha(0.3 + 0.7 * quality)

        # 文献值参考线
        ax.axhline(
            y=literature_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"文献值 ({literature_value} GPa)",
        )

        # 添加数值标签和收敛质量
        for i, (bar, value, quality) in enumerate(
            zip(bars, elastic_constants, convergence_quality, strict=False)
        ):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(height * 0.02, 2),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=12,
                weight="bold",
            )

            # 计算误差
            error = (value - literature_value) / literature_value * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(height * 0.08, 8),
                f"({error:+.0f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                color="gray",
            )

            # 显示收敛率
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height / 2,
                f"{quality:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                weight="bold",
            )

        ax.set_xlabel("剪切模式", fontsize=12)
        ax.set_ylabel("弹性常数 (GPa)", fontsize=12)
        ax.set_title(
            f"弹性常数汇总 - {supercell_size[0]}×{supercell_size[1]}×{supercell_size[2]}系统\n平均值: {np.mean(elastic_constants):.1f} GPa",
            fontsize=14,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(directions)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=10)

        # 设置y轴范围确保能看到所有数据
        max_val = max(max(elastic_constants), literature_value)
        ax.set_ylim(0, max_val * 1.3)
