import os

# 输出文件名
all_file = "scripts/all.txt"
src_file = "scripts/src.txt"
tests_file = "scripts/tests.txt"

# 指定要遍历的文件夹路径
src_folder = "../src"
tests_folder = "../tests"

# 不需要的文件列表，例如CMake生成的文件
exclude_files = ["CMakeCXXCompilerId.cpp", "CMakeFiles", "CMakeCache.txt"]


def write_directory_structure(folder, outfile, all_outfile, indent=""):
    for root, dirs, files in os.walk(folder):
        # 排除 __pycache__ 和 build 文件夹
        dirs[:] = [d for d in dirs if d not in ["__pycache__", "build", "CMakeFiles"]]
        # 只包含 python, lib, cpp 子文件夹
        dirs[:] = [d for d in dirs if d in ["python", "lib", "cpp"]]

        level = root.replace(folder, "").count(os.sep)
        indent = " " * 4 * level
        subindent = " " * 4 * (level + 1)
        outfile.write(f"{indent}{os.path.basename(root)}/\n")
        all_outfile.write(f"{indent}{os.path.basename(root)}/\n")
        for f in files:
            # 排除不需要的文件
            if f not in exclude_files and (
                f.endswith(".cpp") or f.endswith(".py") or f.endswith(".dll")
            ):
                outfile.write(f"{subindent}{f}\n")
                all_outfile.write(f"{subindent}{f}\n")


# 打开文件以追加写入并设置编码为 UTF-8
with open(src_file, "w", encoding="utf-8") as src_outfile, open(
    tests_file, "w", encoding="utf-8"
) as tests_outfile, open(all_file, "w", encoding="utf-8") as all_outfile:

    # 写入 src 文件夹结构
    write_directory_structure(src_folder, src_outfile, all_outfile)

    # 处理 src 文件夹
    for root, _, files in os.walk(src_folder):
        for file in files:
            # 只处理 .cpp 和 .py 文件，并排除不需要的文件
            if file not in exclude_files and (
                file.endswith(".cpp") or file.endswith(".py")
            ):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    src_outfile.write(content)
                    src_outfile.write("\n\n")  # 每个文件间加入换行
                    all_outfile.write(content)
                    all_outfile.write("\n\n")  # 每个文件间加入换行

    # 写入 tests 文件夹结构
    write_directory_structure(tests_folder, tests_outfile, all_outfile)

    # 处理 tests 文件夹
    for root, _, files in os.walk(tests_folder):
        for file in files:
            # 只处理 .cpp 和 .py 文件，并排除不需要的文件
            if file not in exclude_files and (
                file.endswith(".cpp") or file.endswith(".py")
            ):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    tests_outfile.write(content)
                    tests_outfile.write("\n\n")  # 每个文件间加入换行
                    all_outfile.write(content)
                    all_outfile.write("\n\n")  # 每个文件间加入换行
