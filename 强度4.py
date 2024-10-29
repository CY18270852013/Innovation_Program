# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:20:55 2024

@author: Chen Yong
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

# 读取IBTrACS数据
dataset = nc.Dataset(r'C:\Users\Chen Yong\Desktop\Temporary desktop file\Research and cultivation project\IBTrACS.SI.v04r00.nc')

# 提取所需变量数据
usa_pres = dataset.variables['usa_pres'][:, :]
usa_wind = dataset.variables['usa_wind'][:, :]
usa_atcf_id = dataset.variables['usa_atcf_id'][:, :].astype(str)
usa_time = dataset.variables['time'][:]
usa_lat = dataset.variables['usa_lat'][:, :]
usa_lon = dataset.variables['usa_lon'][:, :]

# 定义起始日期和终止日期
start_date = np.datetime64('2023-02-06')
end_date = np.datetime64('2023-03-14')

# 找到目标热带气旋"Freddy"的索引
target_storm_indices = np.where(usa_atcf_id == 'SI112023')[0]
if len(target_storm_indices) > 0:
    target_storm_index = target_storm_indices[0]
    # 提取目标热带气旋的数据
    target_pres = usa_pres[target_storm_index, :]
    target_wind = usa_wind[target_storm_index, :]
    target_time = usa_time

    # 找到起始日期和终止日期的索引
    start_index = np.where(target_time == start_date)[0][0]
    end_index = np.where(target_time == end_date)[0][0]

    # 提取起始日期到终止日期的数据
    target_pres = target_pres[start_index:end_index + 1]
    target_wind = target_wind[start_index:end_index + 1]

    # 绘制强度日变化图
    plt.plot(target_time[start_index:end_index + 1], target_pres, label='Minimum Central Pressure (mb)')
    plt.plot(target_time[start_index:end_index + 1], target_wind, label='Maximum Sustained Wind Speed (kts)')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Intensity')
    plt.title('Intensity Variation of Tropical Cyclone "Freddy" (2023)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("目标热带气旋未找到。")
    