import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation


def extract_data_from_simplified_log(log_file):
    """
    从精简后的日志文件中提取原子位置、能量、最大力和能量变化等信息。
    """
    data = {
        "steps": [],
        "positions": [],
        "energies": [],
        "max_forces": [],
        "energy_changes": [],
    }

    step_pattern = re.compile(r"Step (\d+):")
    position_pattern = re.compile(r"\[\s*([-.0-9e\s,]+)\s*\]")
    energy_pattern = re.compile(r"Total Energy: ([-0-9.e+]+) eV")
    max_force_pattern = re.compile(r"Max Force: ([-0-9.e+]+) eV/Å")
    energy_change_pattern = re.compile(r"Energy Change: ([-0-9.e+]+) eV")

    with open(log_file, "r", encoding="utf-8") as f:
        current_positions = []
        current_step = None
        current_energy = None
        current_max_force = None
        current_energy_change = None

        for line in f:
            # 匹配步骤
            step_match = step_pattern.search(line)
            if step_match:
                if current_step is not None and current_positions:
                    data["steps"].append(current_step)
                    data["positions"].append(np.array(current_positions))
                    data["energies"].append(current_energy)
                    data["max_forces"].append(current_max_force)
                    data["energy_changes"].append(current_energy_change)

                # 重置当前步骤数据
                current_step = int(step_match.group(1))
                current_positions = []
                current_energy = None
                current_max_force = None
                current_energy_change = None

            # 匹配原子位置
            position_match = position_pattern.findall(line)
            if position_match:
                for pos in position_match:
                    pos_list = [float(x) for x in pos.split(",")]
                    current_positions.append(pos_list)

            # 匹配能量、最大力和能量变化
            energy_match = energy_pattern.search(line)
            if energy_match:
                current_energy = float(energy_match.group(1))

            max_force_match = max_force_pattern.search(line)
            if max_force_match:
                current_max_force = float(max_force_match.group(1))

            energy_change_match = energy_change_pattern.search(line)
            if energy_change_match:
                current_energy_change = float(energy_change_match.group(1))

        # 处理最后一个步骤的数据
        if current_step is not None and current_positions:
            data["steps"].append(current_step)
            data["positions"].append(np.array(current_positions))
            data["energies"].append(current_energy)
            data["max_forces"].append(current_max_force)
            data["energy_changes"].append(current_energy_change)

    return data


def update_plot(step, data, scatter, ax):
    """
    更新动画帧的内容。
    """
    ax.cla()  # 清除当前的绘图内容
    pos = data["positions"][step]
    scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Atom positions at Step {data['steps'][step]}")

    # 显示额外信息
    energy = data["energies"][step]
    max_force = data["max_forces"][step]
    energy_change = data["energy_changes"][step]

    ax.text2D(0.05, 0.95, f"Energy: {energy:.6f} eV", transform=ax.transAxes)
    ax.text2D(0.05, 0.90, f"Max Force: {max_force:.6f} eV/Å", transform=ax.transAxes)
    ax.text2D(
        0.05, 0.85, f"Energy Change: {energy_change:.6e} eV", transform=ax.transAxes
    )

    ax.set_xlim([-1, 3])  # 根据你的数据设定合理的x轴范围
    ax.set_ylim([-1, 3])  # 根据你的数据设定合理的y轴范围
    ax.set_zlim([-1, 3])  # 根据你的数据设定合理的z轴范围

    return (scatter,)


def animate_positions(data):
    """
    使用matplotlib的FuncAnimation绘制每100步的轨迹动画。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter([], [], [])  # 初始空的scatter

    total_steps = len(data["positions"])

    # 使用FuncAnimation来创建动画
    ani = FuncAnimation(
        fig,
        update_plot,
        frames=range(total_steps),
        fargs=(data, scatter, ax),
        interval=500,  # 动画每帧之间的间隔（毫秒）
        blit=False,
    )

    plt.show()


if __name__ == "__main__":
    log_file_path = "simplified_log_with_energy.log"  # 替换为你的精简日志文件路径
    data = extract_data_from_simplified_log(log_file_path)
    animate_positions(data)
