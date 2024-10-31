import os
import re

# 输出文件名
all_file = "scripts/all.txt"
src_file = "scripts/src.txt"
tests_file = "scripts/tests.txt"
src_min_file = "scripts/src_min.txt"

# 指定要遍历的文件夹路径
src_folder = "./src"
tests_folder = "./tests"

# 不需要的文件列表，例如CMake生成的文件
exclude_files = ["CMakeCXXCompilerId.cpp", "CMakeFiles", "CMakeCache.txt"]

# 匹配注释的正则表达式
cpp_doc_block_pattern = re.compile(r"/\*\*(?!@file)([\s\S]*?)\*/", re.MULTILINE)
python_doc_block_pattern = re.compile(r'""".*?"""', re.DOTALL)


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


def remove_function_doc_blocks(content, file_extension):
    """
    删除函数和类定义中的块注释，保留文件开头的文档注释

    Parameters
    ----------
    content : str
        文件内容字符串
    file_extension : str
        文件扩展名，用于判断是 Python 还是 C++ 文件

    Returns
    -------
    str
        处理后的文件内容字符串
    """
    if file_extension == ".cpp":
        # 保留文件开头的文件信息块注释，删除其余的块注释
        # 查找第一个块注释，如果包含 @file 则保留
        first_block_match = re.search(r"/\*\*([\s\S]*?)\*/", content)
        if first_block_match and "@file" in first_block_match.group(0):
            # 保留第一个块注释，删除后续块注释
            result = content[: first_block_match.end()] + cpp_doc_block_pattern.sub(
                "", content[first_block_match.end() :]
            )
        else:
            # 没有匹配到 @file 的块注释，则删除所有
            result = cpp_doc_block_pattern.sub("", content)
    elif file_extension == ".py":
        # 保留 Python 文件开头的文件信息注释，删除其余的多行字符串注释
        result = python_doc_block_pattern.sub("", content)
    else:
        result = content

    return result


# 打开文件以追加写入并设置编码为 UTF-8
with open(src_file, "w", encoding="utf-8") as src_outfile, open(
    tests_file, "w", encoding="utf-8"
) as tests_outfile, open(all_file, "w", encoding="utf-8") as all_outfile, open(
    src_min_file, "w", encoding="utf-8"
) as src_min_outfile:

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
                    filtered_content = remove_function_doc_blocks(
                        content, os.path.splitext(file)[1]
                    )
                    src_outfile.write(content + "\n\n")
                    all_outfile.write(content + "\n\n")
                    src_min_outfile.write(filtered_content + "\n\n")

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
                    tests_outfile.write(content + "\n\n")
                    all_outfile.write(content + "\n\n")
