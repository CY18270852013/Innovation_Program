from tqdm import tqdm
import numpy as np
from PIL import Image as img
import os


target_size = (4096, 1)
image_matrices=[]
save_folder=r"/home/lizixuan/matrix/"
subfolders=os.listdir(r"/home/lizixuan/homework/DATA/")
for subfolder in tqdm(subfolders):
    subfolder_path = os.path.join(r"/home/lizixuan/homework/DATA/", subfolder)
    if os.path.isdir(subfolder_path):
        # 遍历子文件夹中的所有文件
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # 打开图片文件
                image_path = os.path.join(subfolder_path, filename)
                if not os.path.exists(image_path):
                   print(f"File '{filename}' does not exist. Skipping...")
                   continue
                image = img.open(image_path)
                imge = image.resize(target_size)
            
            # 将图片转换为灰度图并转换为NumPy数组
                img_array = np.array(imge.convert('L'))
            
            # 将图片矩阵添加到列表中
                image_matrices.append(img_array)
    

 
combined_matrix = np.concatenate(image_matrices, axis=0)
save_path = os.path.join(save_folder, 'matrix.npy')
np.save(save_path, combined_matrix)

