# -*- coding: utf-8 -*-
"""
Created on Sat May  6 21:58:42 2023

@author: Chen Yong
"""

fp1 = open(r'C:\Users\Chen Yong\Desktop\a.txt')
fp2 = open(r'C:\Users\Chen Yong\Desktop\b.txt')
s1 = fp1.read()
s2 = fp2.read()
if s1 == s2:
    print('no difference')
else:
    print('differs')