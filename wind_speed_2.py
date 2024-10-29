# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:32:33 2024

@author: Chen Yong
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 加载ERA5数据
data = xr.open_dataset(r'C:\Users\Chen Yong\Downloads\PV_2007_2016.nc')

# 提取风速数据
wind_speed = data['Wind']

# 提取年份数据
years = data['year'].values.astype(int)

# 提取中国区域的经度和纬度范围
lon_range = slice(70, 140)
lat_range = slice(10, 60)

# 提取指定年份和中国区域的风速数据
selected_data = wind_speed.sel(year=years[(years >= 2007) & (years <= 2022)], month=12, lon=lon_range, lat=lat_range)

# 计算年平均风速
average_wind_speed = selected_data.mean(dim='year')

# 提取经度、纬度和风速数据
lon = average_wind_speed['lon'].values
lat = average_wind_speed['lat'].values
wind = average_wind_speed.values

# 将风速数据中的填充值设置为NaN
wind = np.where(wind == -9999.0, np.nan, wind)

# 绘制空间分布图
plt.figure(figsize=(10, 8))
plt.pcolormesh(lon, lat, wind, cmap='jet', shading='auto')
plt.colorbar(label='Wind Speed (m/s)')
plt.title('Average Wind Speed (2007-2022)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()