import numpy as np
from python.md_simulator import MDSimulator
from python.thermostats import NoseHooverThermostat
from python.barostats import ParrinelloRahmanHooverBarostat
from python.integrators import VelocityVerletIntegrator
from python.mechanics import StressCalculator
from python.structure import Cell, Atom
from python.potentials import EAMAl1Potential
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, Tuple
import os
from datetime import datetime

# 获取当前时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 获取脚本所在的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 创建主输出目录，包含时间戳
output_dir = os.path.join(script_dir, f"simulation_output_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# 创建子目录
logs_dir = os.path.join(output_dir, "logs")
system_states_dir = os.path.join(output_dir, "system_states")
plots_dir = os.path.join(output_dir, "plots")

os.makedirs(logs_dir, exist_ok=True)
os.makedirs(system_states_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# 设置日志文件路径
log_file = os.path.join(logs_dir, f"simulation_debug_{timestamp}.log")
print(f"日志文件将被写入到: {log_file}")  # 打印日志文件的绝对路径

# 配置日志记录
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建文件处理器
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# 创建流处理器
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler.setFormatter(stream_formatter)

# 添加处理器到 logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# 测试日志记录
logger.debug("日志记录器已初始化。")
logger.info("这是一个信息日志。")
logger.error("这是一个错误日志。")

# 其余代码保持不变...


def create_aluminum_fcc(a=4.05):
    """创建FCC铝晶胞"""
    lattice_vectors = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]]) * 4
    basis_positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]

    atoms = []
    for idx, pos in enumerate(basis_positions):
        atoms.append(
            Atom(
                id=idx,
                symbol="Al",
                mass_amu=26.98,
                position=np.array(pos) * a,
                velocity=np.zeros(3),
            )
        )

    cell = Cell(lattice_vectors, atoms)
    return cell.build_supercell((4, 4, 4))


def initialize_velocities(cell: Cell, target_temp: float) -> None:
    """初始化原子速度并移除质心运动"""
    kb = 8.617333262e-5  # eV/K
    np.random.seed(42)

    for atom in cell.atoms:
        sigma = np.sqrt(kb * target_temp / atom.mass)
        atom.velocity = np.random.normal(0, sigma, 3)
        if np.any(np.isnan(atom.velocity)):
            logger.error(f"NaN detected in initial velocity for atom {atom.id}")
            raise ValueError("NaN in initial velocity")
        logger.debug(f"Atom {atom.id} initial velocity: {atom.velocity}")

    # 使用cell的方法移除质心运动
    cell.remove_com_motion()

    # 检查质心运动是否移除成功
    com_velocity = cell.get_com_velocity()
    if np.any(np.isnan(com_velocity)):
        logger.error("NaN detected in center-of-mass velocity")
        raise ValueError("NaN in center-of-mass velocity")
    logger.debug(f"Center-of-mass velocity after removal: {com_velocity}")


def setup_npt_simulation(
    cell: Cell, target_temp: float, target_pressure: float
) -> Tuple[MDSimulator, StressCalculator]:
    """设置NPT模拟"""
    # 设置势能
    potential = EAMAl1Potential(cutoff=6.5)
    logger.debug(f"EAMAl1Potential initialized with cutoff {potential.cutoff}")

    # 设置热浴配置
    thermostat_config = {
        "type": "NoseHoover",
        "params": {
            "target_temperature": target_temp,
            "time_constant": 0.1,
        },
    }

    # 设置压浴配置 - 使用新的压力张量接口
    pressure_tensor = np.eye(3) * target_pressure  # 等静压
    barostat_config = {
        "target_pressure": pressure_tensor,
        "time_constant": 0.2,
        "compressibility": 5e-6,
        "W": None,  # 自动计算晶胞质量
        "Q": np.ones(9) * (0.3**2),  # 如果需要手动设置Q，可以取消注释并确保长度为9
    }

    # 创建应力计算器
    stress_calculator = StressCalculator()

    # 创建压浴实例，传入应力计算器
    barostat = ParrinelloRahmanHooverBarostat(
        target_pressure=barostat_config["target_pressure"],
        time_constant=barostat_config["time_constant"],
        compressibility=barostat_config["compressibility"],
        W=barostat_config["W"],
        # Q=barostat_config["Q"],  # 如果需要手动设置Q
        stress_calculator=stress_calculator,
    )

    # 创建模拟器
    simulator = MDSimulator(
        cell=cell,
        potential=potential,
        integrator=VelocityVerletIntegrator(),
        thermostat=thermostat_config,
        barostat=barostat,
    )
    logger.debug("MDSimulator initialized with ParrinelloRahmanHooverBarostat")

    return simulator, stress_calculator


def calculate_system_state(
    cell: Cell, potential: EAMAl1Potential, stress_calculator: StressCalculator
) -> Dict[str, Any]:
    """计算系统当前状态"""
    state = {}

    # 计算温度和能量
    state["temperature"] = cell.calculate_temperature()
    state["kinetic_energy"] = sum(
        0.5 * atom.mass * np.dot(atom.velocity, atom.velocity) for atom in cell.atoms
    )
    state["potential_energy"] = potential.calculate_energy(cell)
    state["total_energy"] = state["kinetic_energy"] + state["potential_energy"]

    # 计算应力张量和压力
    stress_components = stress_calculator.calculate_total_stress(cell, potential)
    state["stress_components"] = stress_components
    state["pressure"] = {k: np.trace(v) / 3.0 for k, v in stress_components.items()}

    # 计算晶胞参数
    lattice = cell.lattice_vectors
    dimensions = np.sqrt(np.sum(lattice**2, axis=1))
    angles = (
        np.array(
            [
                np.arccos(
                    np.dot(lattice[1], lattice[2]) / (dimensions[1] * dimensions[2])
                ),
                np.arccos(
                    np.dot(lattice[0], lattice[2]) / (dimensions[0] * dimensions[2])
                ),
                np.arccos(
                    np.dot(lattice[0], lattice[1]) / (dimensions[0] * dimensions[1])
                ),
            ]
        )
        * 180
        / np.pi
    )

    state["cell_dimensions"] = dimensions
    state["cell_angles"] = angles
    state["volume"] = cell.volume

    return state


def save_system_state(cell: Cell, step: int):
    """保存系统状态到文件"""
    positions = np.array([atom.position for atom in cell.atoms])
    velocities = np.array([atom.velocity for atom in cell.atoms])
    forces = np.array([atom.force for atom in cell.atoms])

    file_path = os.path.join(system_states_dir, f"system_state_step_{step}.npz")
    np.savez(
        file_path,
        positions=positions,
        velocities=velocities,
        forces=forces,
    )
    logger.info(f"System state at step {step} saved to {file_path}")


def run_npt_simulation(
    target_temp: float = 300.0,  # K
    target_pressure: float = 0.0,  # GPa
    n_steps: int = 500,
    dt: float = 0.005,  # fs
    record_every: int = 5,
) -> Dict[str, Any]:
    """运行NPT系综模拟"""
    # 初始化系统
    cell = create_aluminum_fcc()
    logger.debug(f"Initial cell created: {cell}")
    initialize_velocities(cell, target_temp)
    logger.debug("Velocities initialized")

    # 设置模拟器
    simulator, stress_calculator = setup_npt_simulation(
        cell, target_temp, target_pressure
    )
    logger.debug("Simulator and stress calculator set up")

    # 初始化历史数据记录
    history = {
        "time": [],
        "temperature": [],
        "pressure": {"kinetic": [], "virial": [], "lattice": [], "total": []},
        "energy": {"kinetic": [], "potential": [], "total": []},
        "volume": [],
        "lattice_params": [],
        "stress_tensor": [],
        "cell_dimensions": [],
        "cell_angles": [],
    }

    # 记录初始状态
    logger.info(f"Starting NPT simulation...")
    logger.info(f"Target temperature: {target_temp} K")
    logger.info(f"Target pressure: {target_pressure} GPa")
    logger.info(f"Initial volume: {cell.volume:.3f} Å³")

    # 运行模拟
    for step in range(n_steps):
        try:
            logger.debug(f"Running step {step}")
            # 运行一步模拟
            simulator.run(1, dt)
        except ValueError as e:
            logger.error(f"Simulation stopped at step {step} due to error: {e}")
            save_system_state(cell, step)  # 保存当前系统状态
            break

        if step % record_every == 0:
            current_time = step * dt
            state = calculate_system_state(cell, simulator.potential, stress_calculator)
            logger.debug(f"State at step {step}: {state}")

            # 记录数据
            history["time"].append(current_time)
            history["temperature"].append(state["temperature"])

            for key in ["total"]:
                history["pressure"][key].append(state["pressure"][key])

            history["energy"]["kinetic"].append(state["kinetic_energy"])
            history["energy"]["potential"].append(state["potential_energy"])
            history["energy"]["total"].append(state["total_energy"])

            history["volume"].append(state["volume"])
            history["stress_tensor"].append(state["stress_components"]["total"])
            history["cell_dimensions"].append(state["cell_dimensions"])
            history["cell_angles"].append(state["cell_angles"])

            # 输出状态
            if step % (record_every * 10) == 0:
                logger.info(
                    f"Step {step}: T={state['temperature']:.1f}K, "
                    f"P={state['pressure']['total']:.3f}GPa, "
                    f"V={state['volume']:.1f}Å³"
                )
                logger.info(f"Cell dimensions: {state['cell_dimensions']}")
                logger.info(f"Cell angles: {state['cell_angles']}")
                logger.info(f"Stress tensor:\n{state['stress_components']['total']}")

                # 检查是否有 NaN
                if np.isnan(state["temperature"]) or np.isnan(state["volume"]):
                    logger.error(
                        "Detected NaN values in the system state. Stopping simulation."
                    )
                    save_system_state(cell, step)  # 保存当前系统状态
                    break

                # 额外调试信息
                for atom in cell.atoms:
                    if np.any(np.isnan(atom.position)) or np.any(
                        np.isnan(atom.velocity)
                    ):
                        logger.error(f"Atom {atom.id} has NaN in position or velocity.")
                        save_system_state(cell, step)  # 保存当前系统状态
                        break

    # 确认历史数据被正确记录
    logger.debug(f"Final history data: {history}")

    # 绘制结果
    plot_npt_results(history, target_temp, target_pressure)
    logger.info("Simulation completed and results plotted.")

    return history, cell


def plot_npt_results(
    history: Dict[str, Any], target_temp: float, target_pressure: float
):
    """绘制NPT模拟结果"""
    if not history["time"]:
        logger.error("No data to plot.")
        return

    # 转换为NumPy数组以便于处理
    time = np.array(history["time"])
    temperature = np.array(history["temperature"])
    pressure_total = np.array(history["pressure"]["total"])
    kinetic_energy = np.array(history["energy"]["kinetic"])
    potential_energy = np.array(history["energy"]["potential"])
    total_energy = np.array(history["energy"]["total"])
    volume = np.array(history["volume"])
    stress_tensor = np.array(history["stress_tensor"])
    cell_dims = np.array(history["cell_dimensions"])
    cell_angles = np.array(history["cell_angles"])

    # 添加调试日志以检查数据
    logger.debug(f"Temperature data: {temperature}")
    logger.debug(f"Pressure data: {pressure_total}")
    logger.debug(f"Time data: {time}")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

    # 温度演化
    ax1.plot(time, temperature, label="Temperature")
    ax1.axhline(y=target_temp, color="r", linestyle="--", label="Target")
    ax1.set_xlabel("Time (fs)")
    ax1.set_ylabel("Temperature (K)")
    ax1.set_title("Temperature Evolution")
    ax1.legend()
    ax1.grid(True)

    # 压力演化
    for key in ["total"]:
        ax2.plot(time, pressure_total, label=key.capitalize())
    ax2.axhline(y=target_pressure, color="r", linestyle="--", label="Target")
    ax2.set_xlabel("Time (fs)")
    ax2.set_ylabel("Pressure (GPa)")
    ax2.set_title("Pressure Components")
    ax2.legend()
    ax2.grid(True)

    # 能量演化
    ax3.plot(time, kinetic_energy, label="Kinetic Energy")
    ax3.plot(time, potential_energy, label="Potential Energy")
    ax3.plot(time, total_energy, label="Total Energy")
    ax3.set_xlabel("Time (fs)")
    ax3.set_ylabel("Energy (eV)")
    ax3.set_title("Energy Components")
    ax3.legend()
    ax3.grid(True)

    # 体积演化
    ax4.plot(time, volume, label="Volume")
    ax4.set_xlabel("Time (fs)")
    ax4.set_ylabel("Volume (Å³)")
    ax4.set_title("Volume Evolution")
    ax4.legend()
    ax4.grid(True)

    # 晶格参数演化
    if cell_dims.ndim == 2 and cell_dims.shape[1] >= 3:
        for i in range(3):
            ax5.plot(time, cell_dims[:, i], label=f"a{i+1}")
        ax5.set_xlabel("Time (fs)")
        ax5.set_ylabel("Cell Dimensions (Å)")
        ax5.set_title("Lattice Parameters")
        ax5.legend()
        ax5.grid(True)
    else:
        logger.warning("Insufficient cell_dimensions data for plotting.")

    # 晶胞角度演化
    if cell_angles.ndim == 2 and cell_angles.shape[1] >= 3:
        for i in range(3):
            ax6.plot(time, cell_angles[:, i], label=f"α{i+1}")
        ax6.set_xlabel("Time (fs)")
        ax6.set_ylabel("Cell Angles (degrees)")
        ax6.set_title("Cell Angles")
        ax6.legend()
        ax6.grid(True)
    else:
        logger.warning("Insufficient cell_angles data for plotting.")

    plt.tight_layout()

    # 保存绘图结果到 plots 子目录，并包含时间戳
    plot_file = os.path.join(plots_dir, f"npt_simulation_results_{timestamp}.png")
    plt.savefig(plot_file)
    plt.close()
    logger.info(f"NPT simulation results plotted and saved to {plot_file}")


if __name__ == "__main__":
    history, final_cell = run_npt_simulation(
        target_temp=300.0,  # 300K
        target_pressure=0.02,  # 3GPa
        n_steps=20,  # 增加模拟步数以获得更有意义的数据
        dt=0.01,  # 0.01 fs 时间步长
        record_every=1,  # 每5步记录一次
    )
