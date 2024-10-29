import os

# 定义文件夹路径
folder_path = r"D:\迅雷下载"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 构建文件的旧路径
    old_file_path = os.path.join(folder_path, filename)
    
    # 检查文件名是否符合要求（以“_数字”结尾）
    if filename[-2] == '_' and filename[-1].isdigit():
        # 构建文件的新文件名
        new_filename = filename[:-3]  # 去掉末尾的两个字符
        new_file_path = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"文件 {filename} 已重命名为 {new_filename}")
    else:
        print(f"文件 {filename} 文件名不符合要求，无需重命名")

print("操作完成！")
