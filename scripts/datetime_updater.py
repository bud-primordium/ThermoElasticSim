import os
import re
from datetime import datetime

# 定义需要更新的目录
directories_to_update = ["./src/python", "./src/cpp"]

# 日期格式
date_format = "%Y-%m-%d"

# 正则表达式：Python 文件头注释中的日期格式：# 修改日期: YYYY-MM-DD
python_date_pattern = re.compile(
    r"(^# 修改日期: )(\d{4}-\d{2}-\d{2})(\n)", re.MULTILINE
)

# 正则表达式：C++ 文件头注释中的日期格式：@date YYYY-MM-DD
cpp_date_pattern = re.compile(r"(@date )(\d{4}-\d{2}-\d{2})(\n|\*/)", re.MULTILINE)


def update_file_date(file_path, file_type):
    # 读取文件内容
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 获取当前日期，格式为 YYYY-MM-DD
    current_date = datetime.now().strftime(date_format)  # 使用定义的日期格式

    # 根据文件类型匹配不同的正则表达式
    if file_type == "python":
        pattern = python_date_pattern
    elif file_type == "cpp":
        pattern = cpp_date_pattern
    else:
        return  # 不处理其他文件类型

    # 定义替换函数
    def replacement(match):
        return match.group(1) + current_date + match.group(3)

    # 执行替换
    updated_content = pattern.sub(replacement, content)

    # 检查是否有更新，如果有则写回文件
    if updated_content != content:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(updated_content)
        print(f"Updated date in {file_path}")


def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".py"):
                update_file_date(file_path, "python")
            elif file.endswith(".cpp"):
                update_file_date(file_path, "cpp")


if __name__ == "__main__":
    for directory in directories_to_update:
        process_directory(directory)
    print("Date updating completed.")
