# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:48:14 2024

@author: Chen Yong
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 加载ERA5数据
data = xr.open_dataset(r'C:\Users\Chen Yong\Downloads\PV_2007_2016.nc')



# 提取风速数据
aod = data['AOD']

# 提取年份数据
years = data['year'].values.astype(int)

# 提取中国区域的经度和纬度范围
lon_range = slice(70, 140)
lat_range = slice(10, 60)

# 提取指定年份和中国区域的风速数据
selected_data = aod.sel(year=years[(years >= 2007) & (years <= 2022)], month=12, lon=lon_range, lat=lat_range)


# 计算年平均风速
average_aod = selected_data.mean(dim='year')

# 提取经度、纬度和风速数据
lon = average_aod['lon'].values
lat = average_aod['lat'].values
a = average_aod.values

# 将风速数据中的填充值设置为NaN
a = np.where(a == -9999.0, np.nan, a)

# 创建地图
fig = plt.figure(figsize=(10, 8), dpi=1000)

# 创建colorbar
cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])

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
im = ax.pcolormesh(lon, lat, a, cmap='jet', transform=ccrs.PlateCarree(), antialiased=True)
plt.colorbar(im, label='AOD', cax=cax)
plt.title('AOD annual mean(2007-2022)')

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

# 提取中国陆地的风速数据
selected_data_land = aod.sel(year=years[(years >= 2007) & (years <= 2022)], month=12, lon=lon_range, lat=lat_range)
average_aod_land = selected_data_land.mean(dim='year')
a_land = average_aod_land.values
a_land = np.where(a_land == -9999.0, np.nan, a_land)

# 绘制中国陆地的风速空间分布
ax_inset.pcolormesh(lon, lat, a_land, cmap='jet', transform=ccrs.PlateCarree(), antialiased=True)

# 显示图形
plt.show()