# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:06:21 2024

@author: Chen Yong
"""

import numpy as np
t = True
while t == True:
    img = input("输入你想要搜索的照片：")
    date = np.load(r"C:\Users\Chen Yong\Python codes\date.npz", allow_pickle=True)
    result0 = np.load(r"C:\Users\Chen Yong\Python codes\2022onehot.npy", allow_pickle=True)
    date = list(date['data'])
    num = date.index(img[:-4]+".npz")
    print("你搜索的照片所属类别是：", result0[num])
    change = input('修改：')
    if change == "n":
        continue
    elif change == "q":
        t = False
        print("结束搜索")
    else:
        result0[num] = eval(change)
result1 = result0
print(result0)
np.save(r"C:\Users\Chen Yong\Python codes\result1", result1)