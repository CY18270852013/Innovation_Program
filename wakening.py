# -*- coding: utf-8 -*-
"""
Created on Sat May  6 22:53:39 2023

@author: Chen Yong
"""

with open(r'C:\Users\Chen Yong\Desktop\contact.docx', 'r+') as fp:
    fp.write('Zhang,2301\nZhao,2302\nLi,2304\nSun,2305')
    fp.seek(0)
    name = input()
    for line in fp.readlines():
        if name == line[:len(name)]:
            print(line)
            break
    else:
        print('Not found')