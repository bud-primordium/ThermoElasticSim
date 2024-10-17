import os

# 输出文件名
all_file = "all.txt"

# 指定要遍历的文件夹路径
all_folders = ["./src", "./tests"]

# 打开文件以追加写入并设置编码为 UTF-8
with open(all_file, "w", encoding="utf-8") as outfile:
    for folder in all_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                # 只处理 .cpp 和 .py 文件
                if file.endswith(".cpp") or file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")  # 每个文件间加入换行

src_file = "src.txt"

# 打开文件以追加写入并设置编码为 UTF-8
with open(src_file, "w", encoding="utf-8") as outfile:
    for root, _, files in os.walk("./src"):
        for file in files:
            # 只处理 .cpp 和 .py 文件
            if file.endswith(".cpp") or file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n")  # 每个文件间加入换行
