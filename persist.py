# -*- coding: utf-8 -*-
"""
Created on Sun May  7 07:44:17 2023

@author: Chen Yong
"""

def RLC(s):
    count = 1
    result = []
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result.append(s[i-1] + str(count))
            count = 1
    result.append(s[-1] + str(count))
    return ''.join(result)

with open('file1.txt', 'r') as f:
    s = f.read().strip()
compressed_s = RLC(s)
with open('file2.txt', 'w') as f:
    f.write(compressed_s)
