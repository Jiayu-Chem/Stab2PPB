import os
import sys

def rename_files_to_upper(directory):
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在。")
        return

    print(f"正在处理目录: {os.path.abspath(directory)} ...")
    
    count = 0
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 只处理以 .pdb 结尾的文件 (忽略大小写)
        if filename.lower().endswith(".pdb"):
            # 获取完整的大写文件名
            new_filename = filename[:-4].upper() + filename[-4:]
            
            # 如果文件名已经是大写，跳过
            if filename == new_filename:
                continue

            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # 防止覆盖已存在的同名大写文件
            if os.path.exists(new_path):
                print(f"[跳过] 目标文件已存在: {new_filename}")
                continue

            try:
                os.rename(old_path, new_path)
                print(f"[重命名] {filename} -> {new_filename}")
                count += 1
            except Exception as e:
                print(f"[错误] 无法重命名 {filename}: {e}")

    print(f"-" * 30)
    print(f"完成！共重命名了 {count} 个文件。")

if __name__ == "__main__":
    # 默认处理当前脚本所在目录 (".")
    # 如果你想指定特定目录，可以修改这里，例如: target_dir = "./SSYM/pdbs"
    target_dir = "."
    
    # 也可以通过命令行参数传入路径: python rename_pdbs.py /path/to/pdbs
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]

    rename_files_to_upper(target_dir)