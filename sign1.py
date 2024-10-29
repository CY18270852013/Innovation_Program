# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:46:13 2024

@author: Chen Yong
"""

import numpy as np

t = True
while t:
    img = input("输入你想要搜索的照片：")
    date = np.load(r"C:\Users\Chen Yong\Python codes\date.npz", allow_pickle=True)
    result0 = np.load(r"C:\Users\Chen Yong\Python codes\result1.npy", allow_pickle=True)
    date = list(date['data'])
    num = date.index(img[:] + ".npz")
    print("你搜索的照片所属类别是：", result0[num])
    change = input('修改：')
    if change == "n":
        continue
    elif change == "q":
        t = False
        print("结束搜索")
    else:
        result_copy = np.copy(result0)  # 创建一个新的数组对象来存储修改后的值
        result_copy[num] = eval(change)
        result2 = result_copy
print(result2)
np.save(r"C:\Users\Chen Yong\Python codes\result2", result2)
