import re


def preprocess_log(input_log, output_log, step_interval=100):
    """
    从原始日志文件中提取每100步的原子位置信息、总能量、最大力和能量变化，并生成精简的日志文件。

    参数:
    - input_log: 原始日志文件路径
    - output_log: 生成的新日志文件路径
    - step_interval: 每隔多少步提取一次，默认每100步提取
    """
    step_pattern = re.compile(r"Step (\d+) - Atom positions:")
    position_pattern = re.compile(r"\[\s*([-.0-9e\s]+)\s*\]")
    energy_pattern = re.compile(r"Total Energy = ([-0-9.e+]+) eV")
    force_pattern = re.compile(r"Max force = ([0-9.e-]+) eV/Å")
    energy_change_pattern = re.compile(r"Energy Change = ([0-9.e+-]+) eV")

    with open(input_log, "r", encoding="utf-8") as f_in, open(
        output_log, "w", encoding="utf-8"
    ) as f_out:
        current_positions = []
        current_step = None
        current_energy = None
        current_max_force = None
        current_energy_change = None

        for line in f_in:
            # 匹配步骤信息
            step_match = step_pattern.search(line)
            if step_match:
                # 如果有存储的数据并且符合提取条件，写入文件
                if (
                    current_step is not None
                    and current_step % step_interval == 0
                    and current_positions
                ):
                    f_out.write(f"Step {current_step}:\n")
                    f_out.write("Atom positions:\n[\n")
                    for pos in current_positions:
                        f_out.write(f" [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}],\n")
                    f_out.write("]\n")
                    f_out.write(f"Total Energy: {current_energy} eV\n")
                    f_out.write(f"Max Force: {current_max_force} eV/Å\n")
                    f_out.write(f"Energy Change: {current_energy_change} eV\n\n")

                # 重置存储
                current_step = int(step_match.group(1))
                current_positions = []
                current_energy = None
                current_max_force = None
                current_energy_change = None
            else:
                # 匹配原子位置
                position_match = position_pattern.findall(line)
                if position_match:
                    for pos in position_match:
                        current_positions.append([float(x) for x in pos.split()])

                # 匹配能量、最大力和能量变化
                energy_match = energy_pattern.search(line)
                if energy_match:
                    current_energy = float(energy_match.group(1))
                max_force_match = force_pattern.search(line)
                if max_force_match:
                    current_max_force = float(max_force_match.group(1))
                energy_change_match = energy_change_pattern.search(line)
                if energy_change_match:
                    current_energy_change = float(energy_change_match.group(1))

        # 写入最后一组数据（如果满足条件）
        if (
            current_step is not None
            and current_step % step_interval == 0
            and current_positions
        ):
            f_out.write(f"Step {current_step}:\n")
            f_out.write("Atom positions:\n[\n")
            for pos in current_positions:
                f_out.write(f" [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}],\n")
            f_out.write("]\n")
            f_out.write(f"Total Energy: {current_energy} eV\n")
            f_out.write(f"Max Force: {current_max_force} eV/Å\n")
            f_out.write(f"Energy Change: {current_energy_change} eV\n\n")


if __name__ == "__main__":
    input_log_path = "test_optimizers_20241019_045614.log"  # 替换为你的输入日志路径
    output_log_path = "simplified_log_with_energy.log"  # 生成的精简日志文件路径
    preprocess_log(input_log_path, output_log_path)
