# -*- coding: utf-8 -*-
"""
Created on Sun May  7 13:07:20 2023

@author: Chen Yong
"""

import random
with open(r'C:\Users\Chen Yong\Desktop\c.txt', 'w+') as fp:
    for i in range(100):
        total = 0
        nums_str = ''.join(str(x) for x in random.sample(range(10), 10))
        for ch in nums_str:
            total += int(ch)
        fp.write(nums_str + '->' + str(total) + '\n')
    fp.seek(0)
    nums = fp.readlines()
    for l in nums:
        print(l.strip())

    