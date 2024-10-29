# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:09:36 2024
@author: Chen Yong
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

# 读取IBTrACS数据文件
data = nc.Dataset(r'C:\Users\Chen Yong\Desktop\Temporary desktop file\Research and cultivation project\IBTrACS.ALL.v04r00.nc')

# 获取需要的变量数据
usa_pres = data.variables['usa_pres'][:, 35:45]  # 取2023年数据，日期范围为2月6日到3月14日
usa_wind = data.variables['usa_wind'][:, 35:45]

# 获取时间轴信息
time = data.variables['time'][35:45]
dates = nc.num2date(time, units=data.variables['time'].units)

# 找到"Freddy"对应的索引
storm_names = data.variables['name'][:]
if 'FREDDY' in storm_names:
    freddy_index = np.where(storm_names == 'FREDDY')[0][0]
else:
    freddy_index = -1  # 如果"Freddy"不在数据中，则设置索引为-1

if freddy_index >= 0:
    # 提取"Freddy"的中心气压和最大风速数据
    freddy_pres = usa_pres[freddy_index]
    freddy_wind = usa_wind[freddy_index]

    # 绘制强度日变化图
    plt.figure(figsize=(12, 6))
    plt.plot(dates, freddy_pres, label='Minimum Central Pressure')
    plt.plot(dates, freddy_wind, label='Maximum Sustained Wind Speed')
    plt.xlabel('Date')
    plt.ylabel('Intensity')
    plt.title('Intensity Variation of Tropical Cyclone "FREDDY" (2023)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Tropical Cyclone 'FREDDY' not found in the data.")

# 关闭数据文件
data.close()