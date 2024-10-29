import os
import shutil

# 定义大文件夹路径
source_dir = r"D:\迅雷下载"

# 遍历大文件夹下的所有子文件夹
for root, dirs, files in os.walk(source_dir):
    for directory in dirs:
        # 拼接子文件夹的路径
        subdir_path = os.path.join(root, directory)
        # 获取子文件夹中的文件列表
        print(subdir_path)
        subdir_files = os.listdir(subdir_path)
        #if len(subdir_files) == 1:  # 确保子文件夹下只有一个文件
            # 获取子文件夹中的文件名
        file_name = subdir_files[0]
            # 构建文件的源路径
        src_file = os.path.join(subdir_path, file_name)
            # 构建目标路径，将文件剪切到大文件夹下
        dest_file = os.path.join(root, file_name)
            # 如果目标路径已存在同名文件，先删除
            #if os.path.exists(dest_file):
                #os.remove(dest_file)
            # 剪切文件到目标路径
        shutil.move(src_file, dest_file)
        print(f"文件 {file_name} 已剪切到 {root}")

print("操作完成！")
