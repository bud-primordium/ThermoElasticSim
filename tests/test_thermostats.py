# 文件名: test_thermostats.py

import pytest
import numpy as np
import logging
import os
from datetime import datetime
from python.thermostats import (
    BerendsenThermostat,
    AndersenThermostat,
    NoseHooverThermostat,
    NoseHooverChainThermostat,
)
from python.structure import Atom, Cell
from python.integrators import VelocityVerletIntegrator
from python.potentials import EAMAl1Potential
from python.md_simulator import MDSimulator

# 配置全局日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


@pytest.fixture(scope="session", autouse=True)
def configure_root_logging():
    """配置根日志记录器，确保日志的基本配置"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    yield

    logger.removeHandler(ch)


@pytest.fixture(scope="module")
def log_and_plot_directories(request):
    """为每个测试模块创建日志和绘图目录"""
    module_path = request.node.nodeid.split("::")[0]
    module_name = os.path.splitext(os.path.basename(module_path))[0]

    log_directory = os.path.join("logs", module_name)
    plot_directory = os.path.join(log_directory, f"plots_{current_time}")
    os.makedirs(plot_directory, exist_ok=True)

    # 配置模块日志文件
    log_filename = os.path.join(log_directory, f"{module_name}_{current_time}.log")
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    yield plot_directory

    logger.removeHandler(fh)


@pytest.fixture
def simple_aluminum_cell():
    """创建一个简单的铝原子体系用于测试"""
    lattice = np.array([[4.05, 0.0, 0.0], [0.0, 4.05, 0.0], [0.0, 0.0, 4.05]])

    # 创建2x2x2超胞以提高统计性
    frac_coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 角顶点
            [0.0, 0.5, 0.5],  # 面心
            [0.5, 0.0, 0.5],  # 面心
            [0.5, 0.5, 0.0],  # 面心
        ]
    )

    # 将分数坐标转换为笛卡尔坐标
    cart_coords = np.dot(frac_coords, lattice)

    # 创建原子列表
    atoms = [
        Atom(id=i, symbol="Al", mass_amu=26.98, position=pos)
        for i, pos in enumerate(cart_coords)
    ]
    base_cell = Cell(lattice, atoms)
    supercell = base_cell.build_supercell([2, 2, 2])

    return supercell


def get_temperature(atoms):
    """计算当前温度"""
    kb = 8.617333262e-5  # eV/K
    kinetic = sum(
        0.5 * atom.mass * np.dot(atom.velocity, atom.velocity) for atom in atoms
    )
    dof = 3 * len(atoms) - 3  # 考虑质心运动约束
    return 2.0 * kinetic / (dof * kb)


def initialize_velocities(atoms, temperature, seed=None):
    """改进的Maxwell-Boltzmann分布初始化"""
    if seed is not None:
        np.random.seed(seed)
    kb = 8.617333262e-5

    # 计算总质量和动量
    total_mass = sum(atom.mass for atom in atoms)
    total_momentum = np.zeros(3)

    # 计算速度
    for atom in atoms:
        sigma = np.sqrt(kb * temperature / atom.mass)
        atom.velocity = np.random.normal(0, sigma, 3)
        total_momentum += atom.mass * atom.velocity

    # 移除质心运动
    com_velocity = total_momentum / total_mass
    for atom in atoms:
        atom.velocity -= com_velocity

    # 精确缩放到目标温度
    current_temp = get_temperature(atoms)
    scale = np.sqrt(temperature / current_temp)
    for atom in atoms:
        atom.velocity *= scale


@pytest.mark.parametrize(
    "thermostat_config, steps",
    [
        (
            {
                "type": "Berendsen",
                "params": {"target_temperature": 300.0, "tau": 2.0},
            },
            50000,
        ),
        (
            {
                "type": "Andersen",
                "params": {
                    "target_temperature": 300.0,
                    "collision_frequency": 0.05,
                },
            },
            15000,
        ),
        (
            {
                "type": "NoseHoover",
                "params": {
                    "target_temperature": 300.0,
                    "time_constant": 0.1,
                    "Q": 3.0 * 8.617333262e-5 * 300.0 * 0.1**2,
                },
            },
            10000,
        ),
        (
            {
                "type": "NoseHooverChain",
                "params": {
                    "target_temperature": 300.0,
                    "time_constant": 0.1,
                    "chain_length": 3,
                    "Q": np.array(
                        [3.0 * 8.617333262e-5 * 300.0 * 0.1**2] * 3, dtype=np.float64
                    ),
                },
            },
            20000,
        ),
    ],
)
def test_thermostat_temperature_control(
    request, simple_aluminum_cell, thermostat_config, steps, log_and_plot_directories
):
    """测试恒温器的温度控制效果，并绘制温度演化图"""
    target_temp = thermostat_config["params"]["target_temperature"]

    # 实例化模拟器
    simulator = MDSimulator(
        cell=simple_aluminum_cell,
        potential=EAMAl1Potential(),
        integrator=VelocityVerletIntegrator(),
        thermostat=thermostat_config,
    )

    # 初始化速度
    initialize_velocities(simple_aluminum_cell.atoms, target_temp, seed=42)
    initial_temp = get_temperature(simple_aluminum_cell.atoms)
    thermostat_type = thermostat_config["type"]
    thermostat_params = thermostat_config["params"]
    logger.info(f"Testing {thermostat_type} Thermostat")
    logger.info(f"Initial temperature: {initial_temp:.2f}K")

    # 生成标题和文件名
    title = f"Temperature Evolution - {thermostat_type} with params {thermostat_params}"
    safe_params = (
        "_".join([f"{k}{v}" for k, v in thermostat_params.items() if k != "Q"])
        if thermostat_type != "NoseHooverChain"
        else "_".join([f"{k}{v}" for k, v in thermostat_params.items() if k != "Q"])
    )
    filename = f"{thermostat_type}_params_{safe_params}.png"

    # 运行模拟
    simulator.run(
        steps=steps,
        dt=0.0005,
        plot_title=title,
        plot_filename=filename,
        save_directory=log_and_plot_directories,
    )

    # 产生阶段数据
    temperatures = simulator.temperature[-2000:]
    avg_temp = np.mean(temperatures)
    temp_std = np.std(temperatures)

    logger.info(f"Final temperature: {avg_temp:.2f}K ± {temp_std:.2f}K")

    # 检查动量守恒
    total_momentum = sum(
        atom.mass * atom.velocity for atom in simple_aluminum_cell.atoms
    )
    assert np.allclose(total_momentum, 0, atol=1e-10), "Total momentum not conserved"

    # 检查温度控制
    assert (
        abs(avg_temp - target_temp) / target_temp < 0.1
    ), f"Temperature control failed: {avg_temp:.2f}K ± {temp_std:.2f}K"


def test_nose_hoover_chain_dynamics(simple_aluminum_cell, log_and_plot_directories):
    """测试Nose-Hoover链的动力学特性，并绘制温度演化图"""
    target_temp = 300.0
    thermostat_config = {
        "type": "NoseHooverChain",
        "params": {
            "target_temperature": target_temp,
            "time_constant": 0.1,
            "chain_length": 3,
            "Q": np.array(
                [3.0 * 8.617333262e-5 * 300.0 * 0.1**2] * 3, dtype=np.float64
            ),
        },
    }
    steps = 20000  # 为该恒温器设置特定的模拟步数

    # 实例化模拟器
    simulator = MDSimulator(
        cell=simple_aluminum_cell,
        potential=EAMAl1Potential(),
        integrator=VelocityVerletIntegrator(),
        thermostat=thermostat_config,
    )

    # 初始化速度
    initialize_velocities(simple_aluminum_cell.atoms, target_temp, seed=42)
    initial_temp = get_temperature(simple_aluminum_cell.atoms)
    logger.info(f"Testing NoseHooverChain Thermostat")
    logger.info(f"Initial temperature: {initial_temp:.2f}K")

    # 生成标题和文件名
    thermostat_type = thermostat_config["type"]
    thermostat_params = thermostat_config["params"]
    title = f"Temperature Evolution - {thermostat_type} with params {thermostat_params}"
    safe_params = "_".join(
        [f"{k}{v}" for k, v in thermostat_params.items() if k != "Q"]
    )
    filename = f"{thermostat_type}_params_{safe_params}.png"

    # 运行模拟
    simulator.run(
        steps=steps,
        dt=0.0005,
        plot_title=title,
        plot_filename=filename,
        save_directory=log_and_plot_directories,
    )

    # 记录链变量历史
    thermostat = simulator.thermostat
    if isinstance(thermostat, NoseHooverChainThermostat):
        xi_history = np.array(thermostat.xi_history)
        temp_history = np.array(simulator.temperature)
    else:
        pytest.fail("Thermostat is not NoseHooverChainThermostat")

    # 检查热浴变量是否在波动
    for i in range(thermostat.chain_length):
        if xi_history.shape[1] <= i:
            pytest.fail(f"Chain variable {i} data missing")
        xi_std = np.std(xi_history[:, i])
        assert xi_std > 0, f"Chain variable {i} is not fluctuating"

    # 检查温度是否在合理范围波动
    temp_mean = np.mean(temp_history)
    temp_std = np.std(temp_history)
    assert (
        abs(temp_mean - target_temp) / target_temp < 0.1
    ), f"Temperature control failed: {temp_mean:.2f}K ± {temp_std:.2f}K"
    assert (
        0.01 < temp_std / temp_mean < 0.1
    ), "Temperature fluctuations are outside reasonable range"


def test_energy_conservation(simple_aluminum_cell, log_and_plot_directories):
    """测试NVE系综中的能量守恒，并绘制能量演化图"""
    target_temp = 300.0
    thermostat_config = {
        "type": "NoseHoover",
        "params": {"target_temperature": target_temp, "time_constant": 0.1, "Q": 0.1},
    }
    steps_preheat = 5000
    steps_nve = 10000

    # 实例化模拟器
    simulator = MDSimulator(
        cell=simple_aluminum_cell,
        potential=EAMAl1Potential(),
        integrator=VelocityVerletIntegrator(),
        thermostat=thermostat_config,
    )

    # 初始化速度
    initialize_velocities(simple_aluminum_cell.atoms, target_temp, seed=42)
    initial_temp = get_temperature(simple_aluminum_cell.atoms)
    logger.info(f"Testing Energy Conservation in NVE Ensemble")
    logger.info(f"Initial temperature: {initial_temp:.2f}K")

    # 运行预热阶段
    simulator.run(
        steps=steps_preheat,
        dt=0.0005,
        plot_title="Preheat Phase - NoseHoover Thermostat",
        plot_filename="preheat_nosehoover.png",
        save_directory=log_and_plot_directories,
    )

    # 关闭恒温器，切换到NVE系综
    simulator.thermostat = None

    # 记录初始能量
    initial_energy = simulator.calculate_total_energy()

    # 运行NVE模拟
    energy_history = []
    time_history = []

    for step in range(steps_nve):
        simulator.integrator.integrate(
            simulator.cell, simulator.potential, simulator.dt
        )
        current_energy = simulator.calculate_total_energy()
        energy_history.append(current_energy)
        time_history.append((steps_preheat + step) * simulator.dt)

    energy_history = np.array(energy_history)
    time_history = np.array(time_history)

    # 检查能量守恒
    energy_fluctuation = np.std(energy_history) / np.mean(energy_history)
    assert energy_fluctuation < 1e-4, "Energy is not conserved in NVE ensemble"

    # 绘制能量演化图并保存
    title = "Energy Conservation in NVE Ensemble"
    filename = "Energy_Conservation_NVE.png"

    # 使用模拟器自带的绘图功能（假设存在）
    simulator.plot_energy(
        title=title, filename=filename, save_directory=log_and_plot_directories
    )
