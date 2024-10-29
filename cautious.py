# -*- coding: utf-8 -*-
"""
Created on Sat May  6 23:09:47 2023

@author: Chen Yong
"""

def RLC(s):
    s1 = ''
    count = 0
    while i < len(s):
        if i + 1 < len(s) and s[i + 1] == s[i] and (s1[-2] != s[i] or s1 == ''):
           count += 1
           i += 1
        else:
            s1 = s1 + s[i] + str(count)
            count = 0
            i += 1
    return s1

with open(r'C:\Users\Chen Yong\Desktop\file1.txt', 'r+') as fp1, \ 
    open(r'C:\Users\Chen Yong\Desktop\file2.txt', 'r+') as fp2:
    s = fp1.read()
    s1 = RLC(s)
    fp2.write(s1)