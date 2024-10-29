# -*- coding: utf-8 -*-
"""
Created on Sat May  6 22:30:20 2023

@author: Chen Yong
"""

with open(r'C:\Users\Chen Yong\Desktop\a.txt') as fp1, \
    open(r'C:\Users\Chen Yong\Desktop\b.txt') as fp2: 
        for s1, s2 in zip(fp1.readlines(), fp2.readlines()): 
            if s1 != s2: 
                print('differs') 
                break 
        else: 
            print('no difference')

