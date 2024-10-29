# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:02:15 2024

@author: Chen Yong
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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

# 创建地图
plt.figure(figsize=(10, 8))
m = Basemap(projection='cyl', llcrnrlon=70, llcrnrlat=10, urcrnrlon=140, urcrnrlat=60, resolution='l')

# 绘制地图边界和海岸线
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='lightgray')

# 绘制经纬度网格
m.drawmeridians(np.arange(70, 141, 10), labels=[0, 0, 0, 1], fontsize=10)
m.drawparallels(np.arange(10, 61, 10), labels=[1, 0, 0, 0], fontsize=10)

# 转换经度和纬度坐标为地图投影坐标
lon, lat = np.meshgrid(lon, lat)
x, y = m(lon, lat)

# 绘制空间分布图
plt.pcolormesh(x, y, wind, cmap='jet', shading='auto')
plt.colorbar(label='Wind Speed (m/s)')
plt.title('Average Wind Speed (2007-2022)')
plt.grid(True)

# 添加中国南海小图
south_china_sea_extent = [105, 125, 0, 25]
ax = plt.axes(south_china_sea_extent)
m.drawcoastlines(ax=ax)
m.fillcontinents(color='lightgray', ax=ax)
m.drawparallels(np.arange(0, 26, 5), labels=[1, 0, 0, 0], fontsize=8, ax=ax)
m.drawmeridians(np.arange(105, 126, 5), labels=[0, 0, 0, 1], fontsize=8, ax=ax)

# 显示图形
plt.show()