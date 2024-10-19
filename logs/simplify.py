import re
import os


def find_latest_log(directory, prefix="test_optimizers", extension=".log"):
    """
    在指定目录中查找最新的日志文件

    参数:
    - directory: 日志文件所在的目录
    - prefix: 日志文件的前缀，默认为 'test_optimizers'
    - extension: 日志文件的扩展名，默认为 '.log'

    返回:
    - 最新的日志文件的完整路径
    """
    log_files = [
        f
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(extension)
    ]
    if not log_files:
        raise FileNotFoundError("没有找到匹配的日志文件")

    # 按照修改时间排序，获取最新的文件
    log_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True
    )
    return os.path.join(directory, log_files[0])


def preprocess_log(input_log, output_log, step_interval=100):
    """
    从原始日志文件中提取每100步的原子位置信息、总能量、最大力、能量变化和最近两个原子的距离，
    并生成精简的日志文件。

    参数:
    - input_log: 原始日志文件路径
    - output_log: 生成的新日志文件路径
    - step_interval: 每隔多少步提取一次，默认每100步提取
    """
    patterns = {
        "step": re.compile(r"GD Step (\d+): Max force"),
        "positions": re.compile(
            r"Step \d+ - Atom positions:\n(\[\[([\s\S]+?)\]\])", re.MULTILINE
        ),
        "energy": re.compile(r"Total Energy = ([-0-9.e+]+) eV"),
        "force": re.compile(r"Max force = ([0-9.e-]+) eV/Å"),
        "energy_change": re.compile(r"Energy Change = ([0-9.e+-]+) eV"),
        "distance": re.compile(
            r"Min distance = ([0-9.e-]+) Å between atoms (\d+) and (\d+)"
        ),
    }

    with open(input_log, "r", encoding="utf-8") as f_in, open(
        output_log, "w", encoding="utf-8"
    ) as f_out:
        content = f_in.read()

        # 提取所有匹配项
        steps = patterns["step"].findall(content)
        position_blocks = patterns["positions"].findall(content)
        energies = patterns["energy"].findall(content)
        forces = patterns["force"].findall(content)
        energy_changes = patterns["energy_change"].findall(content)
        distances = patterns["distance"].findall(content)

        for i, step in enumerate(steps):
            current_step = int(step)
            if current_step % step_interval == 0 or current_step == 1:
                f_out.write(f"Step {current_step}:\n")

                # 提取位置信息并格式化输出
                if i < len(position_blocks):
                    # 直接保留原格式输出
                    current_positions_raw = position_blocks[i][0]
                    f_out.write("Atom positions:\n")
                    f_out.write(f"{current_positions_raw}\n")

                # 提取能量、最大力、能量变化并输出
                current_energy = float(energies[i])
                current_max_force = float(forces[i])
                current_energy_change = float(energy_changes[i])
                f_out.write(f"Total Energy: {current_energy} eV\n")
                f_out.write(f"Max Force: {current_max_force} eV/Å\n")
                f_out.write(f"Energy Change: {current_energy_change} eV\n")

                # 提取最近原子的距离并输出
                if i < len(distances):
                    min_distance = float(distances[i][0])
                    atom_pair = (int(distances[i][1]), int(distances[i][2]))
                    f_out.write(
                        f"Min Distance: {min_distance} Å between atoms {atom_pair[0]} and {atom_pair[1]}\n"
                    )
                f_out.write("\n")


if __name__ == "__main__":
    # 找到最新的日志文件
    directory = "."  # 日志文件所在的目录
    input_log_path = find_latest_log(directory)

    # 生成的精简日志文件路径
    output_log_path = "simplified_log_with_energy.log"

    preprocess_log(input_log_path, output_log_path)
    print(f"精简日志已生成: {output_log_path}")
