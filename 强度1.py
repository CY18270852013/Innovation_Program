# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:00:38 2024

@author: Chen Yong
"""


import pandas as pd
import xarray as xr 
import matplotlib.pyplot as plt

# 读取数据
data = xr.open_dataset(r'C:\Users\Chen Yong\Desktop\Temporary desktop file\Research and cultivation project\IBTrACS.SI.v04r00.nc')

# 提取202311S台风弗雷迪的数据
freddy = data[data['usa_agency']=='TCWC Australia'].loc[data['usa_record']=='202311S']

# 提取日期、中心气压和最大风速
date = freddy['time']
pres = freddy['usa_pres']  
wind = freddy['usa_wind']

# 画图
fig, ax1 = plt.subplots()

ax1.plot(date, pres, color='r', label='Central Pressure (mb)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Pressure (mb)', color='r')
ax1.legend(loc='upper left')
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()   
ax2.plot(date, wind, color='b', label='Maximum Wind Speed (kts)')
ax2.set_ylabel('Wind Speed (kts)', color='b') 
ax2.legend(loc='upper right')
ax2.tick_params(axis='y', labelcolor='b')

fig.suptitle('Storm Intensity of Typhoon Freddy (202311S)')

plt.show()