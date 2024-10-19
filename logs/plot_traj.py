import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
from simplify import find_latest_log, preprocess_log


def extract_data_from_simplified_log(log_file):
    """
    从精简后的日志文件中提取原子位置、能量、最大力、能量变化以及最小距离等信息。
    """
    data = {
        "steps": [],
        "positions": [],
        "energies": [],
        "max_forces": [],
        "energy_changes": [],
        "min_distances": [],
        "atom_pairs": [],
    }

    step_pattern = re.compile(r"Step (\d+):")
    position_pattern = re.compile(r"\[\s*([-.0-9e\s,]+)\s*\]")  # 这里的正则表达式不变
    energy_pattern = re.compile(r"Total Energy: ([-0-9.e+]+) eV")
    max_force_pattern = re.compile(r"Max Force: ([-0-9.e+]+) eV/Å")
    energy_change_pattern = re.compile(r"Energy Change: ([-0-9.e+]+) eV")
    min_distance_pattern = re.compile(
        r"Min Distance: ([0-9.e]+) Å between atoms (\d+) and (\d+)"
    )

    with open(log_file, "r", encoding="utf-8") as f:
        current_positions = []
        current_step = None
        current_energy = None
        current_max_force = None
        current_energy_change = None
        current_min_distance = None
        current_atom_pair = None

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
                    data["min_distances"].append(current_min_distance)
                    data["atom_pairs"].append(current_atom_pair)

                # 重置当前步骤数据
                current_step = int(step_match.group(1))
                current_positions = []
                current_energy = None
                current_max_force = None
                current_energy_change = None
                current_min_distance = None
                current_atom_pair = None

            # 匹配原子位置
            position_match = position_pattern.findall(line)
            if position_match:
                for pos in position_match:
                    # 修改这里：使用 split() 以空格分隔
                    pos_list = [float(x) for x in pos.split()]
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

            # 匹配最小距离信息
            min_distance_match = min_distance_pattern.search(line)
            if min_distance_match:
                current_min_distance = float(min_distance_match.group(1))
                current_atom_pair = (
                    int(min_distance_match.group(2)),
                    int(min_distance_match.group(3)),
                )

        # 处理最后一个步骤的数据
        if current_step is not None and current_positions:
            data["steps"].append(current_step)
            data["positions"].append(np.array(current_positions))
            data["energies"].append(current_energy)
            data["max_forces"].append(current_max_force)
            data["energy_changes"].append(current_energy_change)
            data["min_distances"].append(current_min_distance)
            data["atom_pairs"].append(current_atom_pair)

    return data


def update_plot(step, data, scatter, ax):
    """
    更新动画帧的内容
    """
    ax.cla()  # 清除当前的绘图内容
    pos = data["positions"][step]
    scatter = ax.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2], c="blue"
    )  # 普通原子显示为蓝色

    # 获取最近两个原子的索引
    atom_pair = data["atom_pairs"][step]
    if atom_pair:
        # 标红最近的两个原子
        ax.scatter(
            pos[atom_pair[0], 0],
            pos[atom_pair[0], 1],
            pos[atom_pair[0], 2],
            c="red",
            s=100,
        )
        ax.scatter(
            pos[atom_pair[1], 0],
            pos[atom_pair[1], 1],
            pos[atom_pair[1], 2],
            c="red",
            s=100,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Atom positions at Step {data['steps'][step]}")

    # 显示额外信息
    energy = data["energies"][step]
    max_force = data["max_forces"][step]
    energy_change = data["energy_changes"][step]
    min_distance = data["min_distances"][step]

    ax.text2D(0.05, 0.95, f"Energy: {energy:.6f} eV", transform=ax.transAxes)
    ax.text2D(0.05, 0.90, f"Max Force: {max_force:.6f} eV/Å", transform=ax.transAxes)
    ax.text2D(
        0.05, 0.85, f"Energy Change: {energy_change:.6e} eV", transform=ax.transAxes
    )
    if min_distance and atom_pair:
        ax.text2D(
            0.05,
            0.80,
            f"Min Distance: {min_distance:.6f} Å between atoms {atom_pair[0]} and {atom_pair[1]}",
            transform=ax.transAxes,
        )

    # 动态调整 x、y、z 轴范围
    ax.set_xlim([pos[:, 0].min() - 1, pos[:, 0].max() + 1])
    ax.set_ylim([pos[:, 1].min() - 1, pos[:, 1].max() + 1])
    ax.set_zlim([pos[:, 2].min() - 1, pos[:, 2].max() + 1])

    return (scatter,)


def animate_positions(data):
    """
    使用matplotlib的FuncAnimation绘制每100步的轨迹动画
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
    # 找到最新的日志文件
    directory = "."  # 日志文件所在的目录
    input_log_path = find_latest_log(directory)

    # 生成的精简日志文件路径
    output_log_path = "simplified_log_with_energy.log"

    preprocess_log(input_log_path, output_log_path)
    print(f"精简日志已生成: {output_log_path}")
    log_file_path = "simplified_log_with_energy.log"
    data = extract_data_from_simplified_log(log_file_path)
    animate_positions(data)
