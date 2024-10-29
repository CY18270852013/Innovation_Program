# -*- coding: utf-8 -*-
"""
Created on Sun May  7 13:34:12 2023

@author: Chen Yong
"""
import os
dirpath = r'C:\Users\Chen Yong\Desktop'
files = os.listdir(dirpath)
for ch in files:
    if ch.endswith('.txt'):
        filepath = os.path.join(dirpath, ch)
        fp = open(filepath)
        lines = len(fp.readlines())
        print('{0}: {1}'.format(ch, lines))