# 文件名: finite_temp_elasticity.py
# 作者: Gilbert Young
# 修改日期: 2024-12-06
# 文件描述: 实现有限温条件下的弹性常数计算工作流。

import numpy as np
import matplotlib.pyplot as plt
from .md_simulator import MDSimulator
from .thermostats import NoseHooverChainThermostat, AndersenThermostat
from .barostats import (
    ParrinelloRahmanHooverBarostat,
    BerendsenBarostat,
    AndersenBarostat,
)
from .deformation import Deformer
from .mechanics import StressCalculator
from .utils import TensorConverter, EV_TO_GPA
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class FiniteTempElasticityWorkflow:
    """有限温弹性常数计算工作流"""

    def __init__(
        self,
        cell,
        potential,
        temperature,
        integrator,
        deformation_delta=0.1,
        num_steps=10,
        npt_equilibration_steps=10000,
        nvt_sampling_steps=500000,
        time_step=1,
        save_path=f"./output/finite_temp_elasticity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    ):

        self.cell = cell
        self.potential = potential
        self.temperature = temperature
        self.integrator = integrator
        self.deformer = Deformer(deformation_delta, num_steps)
        self.stress_calculator = StressCalculator()

        # MD相关参数
        self.npt_steps = npt_equilibration_steps
        self.nvt_steps = nvt_sampling_steps
        self.dt = time_step

        # 设置保存路径
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        # 初始化恒温器和恒压器
        # self.thermostat = NoseHooverChainThermostat(
        #     target_temperature=temperature, time_constant=0.1, chain_length=3
        # )

        self.thermostat = AndersenThermostat(300, 1)

        # self.barostat = ParrinelloRahmanHooverBarostat(
        #     target_pressure=np.zeros((3, 3)), time_constant=0.1
        # )

        self.barostat = AndersenBarostat(0, 1, 300)

    def run_npt_equilibration(self, cell):
        """在NPT系综下进行平衡"""
        simulator = MDSimulator(
            cell=cell,
            potential=self.potential,
            integrator=self.integrator,
            ensemble="NPT",
            thermostat=self.thermostat,
            barostat=self.barostat,
        )

        simulator.run(self.npt_steps, self.dt)

        # 绘制温度、体积和压力演化图
        plt.figure(figsize=(12, 8))

        plt.subplot(311)
        plt.plot(simulator.time, simulator.temperature)
        plt.axhline(y=self.temperature, color="r", linestyle="--")
        plt.ylabel("Temperature (K)")

        plt.subplot(312)
        plt.plot(simulator.time, simulator.volume)
        plt.ylabel("Volume (Å³)")

        plt.subplot(313)
        plt.plot(simulator.time, simulator.pressure)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Time (fs)")
        plt.ylabel("Pressure (GPa)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "npt_evolution.png"))
        plt.close()

        return cell

    def run_nvt_sampling(self, cell):
        """在NVT系综下采样应力"""
        simulator = MDSimulator(
            cell=cell,
            potential=self.potential,
            integrator=self.integrator,
            ensemble="NVT",
            thermostat=self.thermostat,
        )

        # 初始化应力收集
        stress_history = []

        # 每100步收集一次应力数据
        sampling_interval = 100
        for step in range(0, self.nvt_steps, sampling_interval):
            simulator.run(sampling_interval, self.dt)
            stress = self.stress_calculator.compute_stress(cell, self.potential)
            stress_history.append(stress)

        # 计算平均应力
        avg_stress = np.mean(stress_history, axis=0)
        std_stress = np.std(stress_history, axis=0)

        # 绘制应力演化图
        plt.figure(figsize=(10, 6))
        times = np.arange(0, len(stress_history)) * sampling_interval * self.dt
        for i in range(3):
            plt.plot(times, [s[i, i] for s in stress_history], label=f"σ{i+1}{i+1}")
        plt.xlabel("Time (fs)")
        plt.ylabel("Stress (eV/Å³)")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, "nvt_stress_evolution.png"))
        plt.close()

        return avg_stress, std_stress

    def calculate_elastic_constants(self):
        """计算弹性常数的主流程"""
        strain_data = []
        stress_data = []
        stress_std_data = []

        # 获取变形矩阵列表
        F_list = self.deformer.generate_deformation_matrices()

        for i, F in enumerate(F_list):
            # 创建变形后的晶胞副本
            deformed_cell = self.cell.copy()
            deformed_cell.apply_deformation(F)

            # NPT平衡
            logger.info(f"Running NPT equilibration for deformation {i}")
            equilibrated_cell = self.run_npt_equilibration(deformed_cell)

            # NVT采样
            logger.info(f"Running NVT sampling for deformation {i}")
            avg_stress, std_stress = self.run_nvt_sampling(equilibrated_cell)

            # 计算应变
            strain_tensor = 0.5 * (F + F.T) - np.eye(3)
            strain_voigt = TensorConverter.to_voigt(strain_tensor, tensor_type="strain")
            stress_voigt = TensorConverter.to_voigt(avg_stress, tensor_type="stress")
            stress_std_voigt = TensorConverter.to_voigt(
                std_stress, tensor_type="stress"
            )

            strain_data.append(strain_voigt)
            stress_data.append(stress_voigt)
            stress_std_data.append(stress_std_voigt)

        # 将数据转换为numpy数组
        strain_data = np.array(strain_data)
        stress_data = np.array(stress_data)
        stress_std_data = np.array(stress_std_data)

        # 使用最小二乘法拟合弹性常数
        C, residuals, rank, s = np.linalg.lstsq(strain_data, stress_data, rcond=None)
        C = C.T * EV_TO_GPA  # 转换为GPa

        # 保存结果
        np.savetxt(os.path.join(self.save_path, "elastic_constants_GPa.txt"), C)
        np.savetxt(os.path.join(self.save_path, "strain_data.txt"), strain_data)
        np.savetxt(os.path.join(self.save_path, "stress_data.txt"), stress_data)
        np.savetxt(os.path.join(self.save_path, "stress_std_data.txt"), stress_std_data)

        return C
