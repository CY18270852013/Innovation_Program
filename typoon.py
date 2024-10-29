# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 23:31:20 2024

@author: Chen Yong
"""


import netCDF4 as nc4
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import cartopy.crs as ccrs

path='E://' #文件路径
files= os.listdir(path) #得到文件夹下的所有文件名称

fig=plt.figure(figsize=(20,12)) #设置画布大小
parallels = np.arange(0.,90.,3.) 
meridians = np.arange(0.0,360.,3.)   
ax = plt.axes(projection=ccrs.Robinson())   #设置投影方式
# Set figure extent & ticks
ax.set_extent([100, 150, 5, 50])  #设置纬度范围
# plt.grid(linestyle=':',color='y')#
for file in files:  #按照顺序在 files 里面进行每一个文件的 数据名称 循环读取
    f = open(path+file,'r')   # 打开第一个 dat 文件
    records = f.readlines()   # 读取这个文件里面的所有数据
    f.close()                 # 关闭这个dat文件  
   
    date_t = ''               # 设置一个用来表示空格
    btk_lat = []              # 设置一个空的 待传入数据的纬度
    btk_lon = []              # 设置一个空的 待传入数据的经度
    btk_vmax = []             # 风速最大值 Maximum sustained wind speed in knots: 0 - 300 kts.
    btk_time = []             # 时间
    btk_rmw = []              # 最大风速半径 radius of max winds, 0 - 999 n mi.
    btk_name = []             # 台风名称

    for rcd in records:       # 对这个dat文件里面，已经读取的每一行数据进行循环处理
        
        strs = rcd.split(',') #将每一个','分隔开
        if(len(strs)<21):     #判断语句，如果这个被分割开的字符 长度<21 ,继续进行处理
            continue
        date_str = strs[2].strip(' ') #将strs这个list的索引为2的值赋给data_str,既年月日时
        if date_str == date_t:#判读如果是一个空格值，赋给data——str
            continue
        dt = datetime.datetime(int(date_str[0:4]),int(date_str[4:6]),int(date_str[6:8]),\
        int(date_str[8:]),0,0,0)
        btk_time.append(nc4.date2num(dt,units='second since 1970-1-1 00:00:00'))#计算距离给的时间有多少秒，并从后往前排列
        #处理纬度
        lat_str = strs[6].strip()
        #判断南北纬
        if lat_str[-1] == 'N':
            lat_t = float(lat_str[0:-1])*0.1
        else:
            lat_t = float(lat_str[0:-1])*-0.1
        btk_lat.append(lat_t)
        #处理经度
        lon_str = strs[7].strip()
        #判断 东西经
        if lon_str[-1] == 'E':
            lon_t = float(lon_str[0:-1])*0.1
        else:
            lon_t = float(lon_str[0:-1])*-0.1
        btk_lon.append(lon_t)        
        #处理最大风速
        vmax = strs[8].strip()
        btk_vmax.append(float(vmax))#转换为单浮点型，（带小数点）
        #时间
        date_t = date_str
        #最大风速半径
        rmw = strs[19].strip()
        btk_rmw.append(float(rmw))
        #处理台风名称
        if(len(strs) < 27):
            btk_name.append('noname')
        else:
            name = strs[27].strip()
            btk_name.append(name)
#==============================================================================
    btk_lat = np.array(btk_lat) #将得到的list 值转换为数组型的值，为了便于绘图。因为绘图的横纵坐标都是数组排列
    btk_lon = np.array(btk_lon)%360 #因为原始经度为-180 - 0 -180 ，出现断隔，为解决问题，化为 0-360
    btk_time = np.array(btk_time)  #时间转换
    btk_vmax = np.array(btk_vmax)*0.5144 #风速换算公式
    btk_rmw = np.array(btk_rmw)*1.852 #
    #判断，如果数组纬度的值是0，则为nan值，既无法计算的值（无穷大，，），否则即为台风的名称
    if(len(btk_lat) == 0): 
        tc_name = 'noname'
    else:
        index = btk_vmax.argmax()
        tc_name = btk_name[index]
    
    #进行绘图，经度、纬度曲线
    ax.plot(btk_lon,btk_lat,color='k',linewidth=0.5,transform=ccrs.PlateCarree())
    #散点图绘制，经度、纬度、最大风速，
    cb = ax.scatter(btk_lon,btk_lat,c=btk_vmax,s=10.0,transform=ccrs.PlateCarree()
                    ,vmin=10,vmax=60)
ax.coastlines()
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

plt.colorbar(cb,label='Vmax (m/s)',pad=0.07,orientation='vertical',shrink=1)
plt.title('  path')
# 保存绘制图片 ，注意保存路径不能放在dat文件夹中     
#fig.savefig(path2+'tester.tiff',format='tiff',dpi=100)

