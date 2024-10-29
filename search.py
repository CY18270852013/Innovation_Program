import numpy as np
import pickle
with open("/home/lizixuan/matrix/namelist.pkl", 'rb') as f:
    matrix_from_pkl = pickle.load(f)
target_value = input("请输入要查找的值: ")

indices = matrix_from_pkl.index(target_value)
print(indices)
matrix_from_npy=np.load("/home/lizixuan/matrix/cluster_labels.npy")
print(matrix_from_npy[indices])
