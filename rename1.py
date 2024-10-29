# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:12:09 2024

@author: Chen Yong
"""

import os

# 定义文件夹路径
folder_path = r"E:\电影原片"

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 构建文件的旧路径
    old_file_path = os.path.join(folder_path, filename)
    
    # 检查文件名是否符合要求
    if len(filename) > 2:
        # 构建文件的新文件名
        new_filename = filename[:-2] + ".mp4"  # 删除末尾的两个字符并加上.mp4后缀
        new_file_path = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"文件 {filename} 已重命名为 {new_filename}")
    else:
        print(f"文件 {filename} 文件名长度不足，无法重命名")

print("操作完成！")
