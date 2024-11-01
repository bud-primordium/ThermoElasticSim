# md_simulator.py
# 作者: Gilbert Young
# 修改日期: 2024-11-01
# 文件描述: 实现分子动力学模拟器 MDSimulator 类，用于执行分子动力学模拟。

import matplotlib.pyplot as plt
import os
import numpy as np
import logging
from typing import Optional, Dict

from .thermostats import (
    Thermostat,
    BerendsenThermostat,
    AndersenThermostat,
    NoseHooverThermostat,
    NoseHooverChainThermostat,
)

logger = logging.getLogger(__name__)


class MDSimulator:
    """
    分子动力学模拟器类

    Parameters
    ----------
    cell : Cell
        包含原子的晶胞对象
    potential : Potential
        势能对象，用于计算作用力
    integrator : Integrator
        积分器对象，用于时间推进模拟
    thermostat : dict or Thermostat, optional
        恒温器配置字典或恒温器对象，用于控制温度
    barostat : Barostat, optional
        压强控制器对象，用于控制压强
    """

    def __init__(
        self,
        cell,
        potential,
        integrator,
        thermostat: Optional[Dict] = None,
        barostat=None,
    ):
        """初始化 MDSimulator 对象"""
        self.cell = cell
        self.potential = potential
        self.integrator = integrator
        self.barostat = barostat  # 为 NPT 系综预留
        self.dt = None

        # 轨迹数据存储（使用列表收集，后续可转换为 NumPy 数组）
        self.time = []
        self.temperature = []
        self.energy = []

        if isinstance(thermostat, dict):
            thermostat_type = thermostat.get("type")
            thermostat_params = thermostat.get("params", {})
            if thermostat_type == "Berendsen":
                self.thermostat = BerendsenThermostat(
                    target_temperature=thermostat_params.get(
                        "target_temperature", 300.0
                    ),
                    tau=thermostat_params["tau"],
                )
            elif thermostat_type == "Andersen":
                self.thermostat = AndersenThermostat(
                    target_temperature=thermostat_params.get(
                        "target_temperature", 300.0
                    ),
                    collision_frequency=thermostat_params["collision_frequency"],
                )
            elif thermostat_type == "NoseHoover":
                self.thermostat = NoseHooverThermostat(
                    target_temperature=thermostat_params.get(
                        "target_temperature", 300.0
                    ),
                    time_constant=thermostat_params["time_constant"],
                    Q=thermostat_params.get("Q", None),
                )
            elif thermostat_type == "NoseHooverChain":
                self.thermostat = NoseHooverChainThermostat(
                    target_temperature=thermostat_params.get(
                        "target_temperature", 300.0
                    ),
                    time_constant=thermostat_params["time_constant"],
                    chain_length=thermostat_params.get("chain_length", 3),
                    Q=thermostat_params.get("Q", None),  # 支持外部传入 Q
                )
            else:
                raise ValueError(f"Unsupported thermostat type: {thermostat_type}")
        else:
            self.thermostat = thermostat

        self.barostat = barostat  # 为 NPT 系综预留

        # 初始化数据收集
        self.time = []
        self.temperature = []
        self.energy = []

    def run(
        self,
        steps: int,
        dt: float,
        plot_title: str = "Temperature Evolution",
        plot_filename: str = "temperature_evolution.png",
        save_directory: str = "./plots/",
    ):
        """
        运行分子动力学模拟

        Parameters
        ----------
        steps : int
            模拟步数
        dt : float
            时间步长
        plot_title : str
            温度演化图的标题
        plot_filename : str
            保存的图像文件名
        save_directory : str
            图像保存的目录
        """
        # 初始化力
        self.potential.calculate_forces(self.cell)
        current_time = 0.0
        self.dt = dt

        for step in range(steps):
            # 积分更新位置和速度
            self.integrator.integrate(self.cell, self.potential, dt)
            current_time += dt

            # 应用恒温器（如果存在）
            if self.thermostat is not None:
                self.thermostat.apply(self.cell.atoms, dt)

            # 应用压强控制器（如果存在）
            if self.barostat is not None:
                self.barostat.apply(self.cell, self.potential, dt)

            # 记录温度
            temp = self.cell.calculate_temperature()
            self.time = np.append(self.time, current_time)
            self.temperature = np.append(self.temperature, temp)

            # 记录能量
            total_energy = self.calculate_total_energy()
            self.energy = np.append(self.energy, total_energy)

            # 数据收集日志（每1000步）
            if (step + 1) % 1000 == 0 or step == steps - 1:
                logger.info(
                    f"MD Step {step + 1}/{steps} completed. "
                    f"Time: {current_time:.2f} fs, Temperature: {temp:.2f} K, Total Energy: {total_energy:.2f} eV"
                )
                logger.debug(f"Temperature history: {self.temperature[-1000:]}")
                logger.debug(f"Energy history: {self.energy[-1000:]}")
        # 转换为 NumPy 数组以提高绘图和分析效率
        self.time = np.array(self.time, dtype=np.float64)
        self.temperature = np.array(self.temperature, dtype=np.float64)
        self.energy = np.array(self.energy, dtype=np.float64)

        # 绘制温度演化图
        self.plot_temperature(plot_title, plot_filename, save_directory)

    def plot_temperature(self, title: str, filename: str, save_directory: str):
        """
        绘制温度随时间的演化图并保存。

        Parameters
        ----------
        title : str
            图表标题。
        filename : str
            保存的文件名。
        save_directory : str
            保存的目录。
        """
        os.makedirs(save_directory, exist_ok=True)
        filepath = os.path.join(save_directory, filename)

        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.temperature, label="Temperature (K)")
        if self.thermostat is not None:
            plt.axhline(
                y=self.thermostat.target_temperature,
                color="r",
                linestyle="--",
                label="Target Temperature",
            )
        plt.xlabel("Time (fs)")
        plt.ylabel("Temperature (K)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Temperature evolution plot saved to {filepath}")

    def plot_energy(self, title: str, filename: str, save_directory: str):
        """
        绘制能量随时间的演化图并保存。

        Parameters
        ----------
        title : str
            图表标题。
        filename : str
            保存的文件名。
        save_directory : str
            保存的目录。
        """
        os.makedirs(save_directory, exist_ok=True)
        filepath = os.path.join(save_directory, filename)

        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.energy, label="Total Energy (eV)")
        plt.xlabel("Time (fs)")
        plt.ylabel("Energy (eV)")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Energy evolution plot saved to {filepath}")

    def calculate_total_energy(self) -> float:
        """
        计算系统的总能量（动能 + 势能）

        Returns
        -------
        float
            系统的总能量，单位eV
        """
        kinetic = self.integrator.calculate_kinetic_energy(self.cell.atoms)
        potential = self.potential.calculate_energy(self.cell)
        return kinetic + potential
