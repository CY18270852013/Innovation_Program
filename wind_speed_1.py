# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:45:56 2024

@author: Chen Yong
"""

import xarray as xr
import matplotlib.pyplot as plt

# 加载ERA5数据
data = xr.open_dataset(r'C:\Users\Chen Yong\Downloads\PV_2007_2016.nc')

# 提取风速数据
wind_speed = data['Wind']

# 计算2007年到2016年的年平均风速
average_wind_speed = wind_speed.sel(year=slice(2007, 2016)).mean(dim='year')

# 绘制空间分布图
average_wind_speed.plot.imshow(x='lon', y='lat', cmap='jet', vmin=0, vmax=20)
plt.title('Average Wind Speed (2007-2016)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()