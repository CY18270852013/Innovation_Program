# -*- coding: utf-8 -*-
"""
Created on Sun May  7 08:48:11 2023

@author: Chen Yong
"""

with open(r'C:\Users\Chen Yong\Desktop\a.txt', 'w') as fp:
    fp.write("How many roads must a man walk down\n"
             "Before you can call him a man\n" 
             "How many seas must a white dove sail\n" 
             "Before she sleeps in the sand\n" 
             "How many times must the cannon balls fly\n" 
             "Before they're forever banned\n" 
             "The answer my friend is blowing in the wind\n" 
             "The answer is blowing in the wind\n")
 
with open(r'C:\Users\Chen Yong\Desktop\a.txt', 'r+') as fp:
    lst = fp.readlines()
    lst.insert(0, "Blowin'in the wind\n")
    lst.insert(1, 'Bob Dylan\n')
    lst.append('1962 by Warner Bros. Inc.\n')
    fp.seek(0)
    fp.write(''.join(lst))
    fp.seek(0)
    lyrics = fp.read()
    print(lyrics)