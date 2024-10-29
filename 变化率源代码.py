# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:59:43 2024

@author: Lenovo
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 加载ERA5数据
data = xr.open_dataset(r'C:\Users\Chen Yong\Downloads\PV_2007_2016.nc')



# 提取风速数据
wind_speed = data['Wind']

# 提取年份数据
years = data['year'].values.astype(int)
months=data['month'].values.astype(int)
# 提取中国区域的经度和纬度范围
lon_range = slice(70, 140)
lat_range = slice(10, 60)

# 提取指定年份和中国区域的风速数据
selected_data = wind_speed.sel(year=years, month=months, lon=lon_range, lat=lat_range)

# 提取2007年和2022年的风速数据
selected_data_2007 = selected_data.sel(year=2007)
selected_data_2022 = selected_data.sel(year=2022)

# 计算风速变化率
change_wind_speed = (selected_data_2022.sum(dim='month') - selected_data_2007.sum(dim='month')) / 1.6

# 提取经度、纬度和变化率数据
lon = data['lon'].values
lat = data['lat'].values
wind = change_wind_speed.values
# 将风速数据中的填充值设置为NaN
wind = np.where(wind == -9999.0, np.nan, wind)

# 创建地图
fig = plt.figure(figsize=(10, 8), dpi=300)

# 创建colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

# 绘制主地图
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 绘制中国边境线
ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)

# 绘制地图边界、海岸线和陆地
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.LAND, color='lightgray')

# 绘制经纬度网格线
ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

# 绘制空间分布图
im = ax.pcolormesh(lon, lat, wind, cmap='jet', transform=ccrs.PlateCarree(), antialiased=True)
plt.colorbar(im, label='Rate of change of wind speed per decade', cax=cax)
plt.title('Trend of wind speed from 2007 to 2022')

# 添加中国南海小图
south_china_sea_extent = [105, 125, 0, 25]
ax_inset = fig.add_axes([0.7, 0.2, 0.2, 0.2], projection=ccrs.PlateCarree())
ax_inset.coastlines(resolution='10m')
ax_inset.add_feature(cfeature.LAND, color='lightgray')
ax_inset.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)

# 绘制中国领海边境线
china_boundaries = cfeature.NaturalEarthFeature(category='cultural',
                                               name='admin_0_boundary_lines_land',
                                               scale='10m',
                                               facecolor='none')
ax_inset.add_feature(china_boundaries, linestyle='-', edgecolor='black')

ax_inset.set_extent(south_china_sea_extent)

# 提取指定年份和中国区域的风速数据
selected_data = wind_speed.sel(year=years, month=months, lon=lon_range, lat=lat_range)

# 提取2007年和2022年的风速数据
selected_data_2007 = selected_data.sel(year=2007)
selected_data_2022 = selected_data.sel(year=2022)

# 计算风速变化率
change_wind_speed = (selected_data_2022.sum(dim='month') - selected_data_2007.sum(dim='month')) / 1.6

wind_land = change_wind_speed.values
wind_land = np.where(wind_land == -9999.0, np.nan, wind_land)

# 绘制中国陆地的风速空间分布
ax_inset.pcolormesh(lon, lat, wind_land, cmap='jet', transform=ccrs.PlateCarree(), antialiased=True)
# 显示图形
plt.show()