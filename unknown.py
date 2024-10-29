from sklearn.cluster import KMeans
import numpy as np
from PIL import Image as img
import os
import matplotlib.pyplot as plt
from sklearn.externals import joblib
target_size = (4096, 1)
image_matrices = []
save_folder=r'E:\result'
subfolders=os.listdir(r'E:\datapartition')
for subfolder in subfolders:
    subfolder_path = os.path.join(r'E:\datapartition', subfolder)
    if os.path.isdir(subfolder_path):
        # 遍历子文件夹中的所有文件
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # 打开图片文件
                image_path = os.path.join(subfolder_path, filename)
                image = img.open(image_path)
                imge = image.resize(target_size)
            
            # 将图片转换为灰度图并转换为NumPy数组
                img_array = np.array(imge.convert('L'))
            
            # 将图片矩阵添加到列表中
                image_matrices.append(img_array)
    

 
combined_matrix = np.concatenate(image_matrices, axis=0)
save_path = os.path.join(save_folder, 'combined_matrix.npy')
np.save(save_path, combined_matrix)
x=combined_matrix
kmeans = KMeans(n_clusters=4, n_init=41)
model = kmeans.fit(x)
joblib.dump(model,'E:/kmeans_model.pkl')
# Load the model
#loaded_model = joblib.load('E:/kmeans_model.pkl')
centers = kmeans.cluster_centers_
labels = kmeans.labels_
save_path1=os.path.join(save_folder,'labels.npy')

np.save(save_path1,labels)
print(centers)
print(labels)
data = np.load(save_path1)

# 统计数据
unique, counts = np.unique(data, return_counts=True)

# 绘制柱状图
plt.bar(unique, counts)
plt.xlabel('Unique Values')
plt.ylabel('Counts')
plt.title('Value Counts')
plt.show()
