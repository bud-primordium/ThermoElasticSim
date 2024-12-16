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
    分子动力学模拟器，支持NVE、NVT和NPT系综

    Parameters
    ----------
    cell : Cell
        包含原子的晶胞对象
    potential : Potential
        势能对象
    integrator : Integrator
        积分器对象
    ensemble : str
        系综类型: 'NVE', 'NVT', 或 'NPT'
    thermostat : Optional[Dict] or Thermostat
        恒温器配置或对象
    barostat : Optional[Barostat]
        恒压器对象
    """

    def __init__(
        self,
        cell,
        potential,
        integrator,
        ensemble="NVE",
        thermostat=None,
        barostat=None,
    ):
        self.cell = cell
        self.potential = potential
        self.integrator = integrator
        self.ensemble = ensemble.upper()
        self.dt = None

        # 验证系综配置
        if self.ensemble not in ["NVE", "NVT", "NPT"]:
            raise ValueError(f"Unknown ensemble: {self.ensemble}")

        if self.ensemble == "NVE" and (thermostat or barostat):
            raise ValueError("NVE ensemble should not have thermostat or barostat")

        if self.ensemble == "NVT" and barostat:
            raise ValueError("NVT ensemble should not have barostat")

        if self.ensemble == "NPT" and not (thermostat and barostat):
            raise ValueError("NPT ensemble requires both thermostat and barostat")

        # 配置恒温器
        if isinstance(thermostat, dict):
            self.thermostat = self._setup_thermostat(thermostat)
        else:
            self.thermostat = thermostat

        self.barostat = barostat

        # 轨迹数据
        self.time = []
        self.temperature = []
        self.energy = []
        self.pressure = []
        self.volume = []

    def _setup_thermostat(self, config):
        """根据配置字典设置恒温器"""
        thermostat_type = config.get("type")
        params = config.get("params", {})

        if thermostat_type == "Berendsen":
            return BerendsenThermostat(
                target_temperature=params.get("target_temperature", 300.0),
                tau=params["tau"],
            )
        elif thermostat_type == "Andersen":
            return AndersenThermostat(
                target_temperature=params.get("target_temperature", 300.0),
                collision_frequency=params["collision_frequency"],
            )
        elif thermostat_type == "NoseHoover":
            return NoseHooverThermostat(
                target_temperature=params.get("target_temperature", 300.0),
                time_constant=params["time_constant"],
                Q=params.get("Q", None),
            )
        elif thermostat_type == "NoseHooverChain":
            return NoseHooverChainThermostat(
                target_temperature=params.get("target_temperature", 300.0),
                time_constant=params["time_constant"],
                chain_length=params.get("chain_length", 3),
                Q=params.get("Q", None),
            )
        else:
            raise ValueError(f"Unsupported thermostat type: {thermostat_type}")

    def run_nve(self, steps, dt):
        """运行NVE系综模拟"""
        current_time = 0.0
        self.dt = dt

        # 初始化力
        self.potential.calculate_forces(self.cell)

        for step in range(steps):
            # 积分更新位置和速度
            self.integrator.integrate(self.cell, self.potential, dt)
            current_time += dt

            # 记录数据
            self._record_state(current_time)

    def run_nvt(self, steps, dt):
        """运行NVT系综模拟"""
        current_time = 0.0
        self.dt = dt

        # 初始化力
        self.potential.calculate_forces(self.cell)

        for step in range(steps):
            # 积分更新位置和速度
            self.integrator.integrate(self.cell, self.potential, dt)

            # 应用恒温器 (修改处：将原来的 apply(self.cell.atoms, dt) 改为 apply(self.cell, dt, self.potential))
            self.thermostat.apply(self.cell, dt, self.potential)

            current_time += dt

            # 记录数据
            self._record_state(current_time)

    def run_npt(self, steps, dt):
        """运行NPT系综模拟"""
        current_time = 0.0
        self.dt = dt

        # 初始化力
        self.potential.calculate_forces(self.cell)

        for step in range(steps):
            # 更新位置和速度(1/4步)
            self.integrator.integrate(self.cell, self.potential, dt / 4)

            # 应用恒压器(1/2步)
            self.barostat.apply(self.cell, dt / 2, self.potential)

            # 应用恒温器(1/2步)
            self.thermostat.apply(self.cell, dt / 2, self.potential)

            # 更新位置和速度(1/2步)
            self.integrator.integrate(self.cell, self.potential, dt / 2)

            # 应用恒温器(1/2步)
            self.thermostat.apply(self.cell, dt / 2, self.potential)

            # 应用恒压器(1/2步)
            self.barostat.apply(self.cell, dt / 2, self.potential)

            # 更新位置和速度(1/4步)
            self.integrator.integrate(self.cell, self.potential, dt / 4)

            current_time += dt

            # 记录数据
            self._record_state(current_time)

    def run(self, steps: int, dt: float):
        """
        运行分子动力学模拟

        Parameters
        ----------
        steps : int
            模拟步数
        dt : float
            时间步长
        """
        if self.ensemble == "NVE":
            self.run_nve(steps, dt)
        elif self.ensemble == "NVT":
            self.run_nvt(steps, dt)
        elif self.ensemble == "NPT":
            self.run_npt(steps, dt)

        # 完成后绘制温度演化图
        self.plot_temperature()

    def _record_state(self, current_time):
        """记录系统状态"""
        temp = self.cell.calculate_temperature()
        total_energy = self.calculate_total_energy()
        volume = self.cell.volume

        self.time.append(current_time)
        self.temperature.append(temp)
        self.energy.append(total_energy)
        self.volume.append(volume)

        if self.ensemble == "NPT":
            pressure = self.barostat.calculate_internal_pressure(
                self.cell.calculate_stress_tensor(self.potential)
            )
            self.pressure.append(pressure)

    def calculate_total_energy(self) -> float:
        """计算系统总能量"""
        kinetic = self.integrator.calculate_kinetic_energy(self.cell.atoms)
        potential = self.potential.calculate_energy(self.cell)
        return kinetic + potential

    def plot_temperature(self):
        """绘制温度演化图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time, self.temperature, label="Temperature (K)")

        if self.thermostat:
            plt.axhline(
                y=self.thermostat.target_temperature,
                color="r",
                linestyle="--",
                label="Target Temperature",
            )

        plt.xlabel("Time (fs)")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature Evolution")
        plt.legend()
        plt.grid(True)
        plt.show()
