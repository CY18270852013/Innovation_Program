import os
import shutil

# 设置源目录和目标目录
src_dir = r"E:\电影原片"
dst_dir = r"E:\电影原片"

# 遍历源目录下的所有子目录
for subdir in os.listdir(src_dir):
    subdir_path = os.path.join(src_dir, subdir)
    # 检查是否为目录
    if os.path.isdir(subdir_path):
        # 获取子目录下的文件
        file_list = os.listdir(subdir_path)
        if file_list:
            # 获取子目录名作为文件名
            file_name = os.path.basename(subdir_path)
            # 构建目标文件路径
            dst_file_path = os.path.join(dst_dir, file_name)
            # 如果目标文件已存在,添加序号
            if os.path.exists(dst_file_path):
                i = 1
                while os.path.exists(os.path.join(dst_dir, f"{file_name}_{i}")):
                    i += 1
                dst_file_path = os.path.join(dst_dir, f"{file_name}_{i}")
            # 将文件剪切到目标目录
            shutil.move(os.path.join(subdir_path, file_list[0]), dst_file_path)
        # 删除空的子目录
        os.rmdir(subdir_path)