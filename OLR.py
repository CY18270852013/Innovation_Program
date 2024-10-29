# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:44:39 2024

@author: Chen Yong
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# 读取netCDF文件
data = nc.Dataset('C:\\Users\\Chen Yong\\Desktop\\Temporary desktop file\\Research and cultivation project\\olr-daily_v01r02-preliminary_20230101_20231219.nc')

# 获取经度、纬度和时间的数据
lon = data.variables['lon'][:]
lat = data.variables['lat'][:]
time = data.variables['time'][:]

# 获取OLR数据
olr = data.variables['olr'][:]

# 计算南北纬5°平均OLR数据
avg_olr = np.mean(olr[:, lat >= -5, :], axis=1)

# 创建一个Hovmuller图
plt.imshow(avg_olr.T, origin='lower', aspect='auto', cmap='jet')

# 设置坐标轴
plt.xlabel('时间')
plt.ylabel('经度')
plt.xticks(range(len(time)), time, rotation='vertical')
plt.yticks(range(len(lon)), lon)

# 添加颜色条
plt.colorbar(label='OLR')

# 显示图像
plt.show()