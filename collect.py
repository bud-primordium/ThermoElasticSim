import os

# 输出文件名
output_file = "output.txt"

# 指定要遍历的文件夹路径
folders = ["./src", "./tests"]

# 打开文件以追加写入并设置编码为 UTF-8
with open(output_file, "w", encoding="utf-8") as outfile:
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                # 只处理 .cpp 和 .py 文件
                if file.endswith(".cpp") or file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n")  # 每个文件间加入换行
