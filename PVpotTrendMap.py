# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:55:56 2024

@author: Chen Yong
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm

# 读取 NetCDF 数据文件
file_path = r'C:\Users\Chen Yong\Desktop\InnovationProgram\PV_2007_2016.nc'
data = xr.open_dataset(file_path)

# 提取PV值
pv_data = data['PV']

# 提取经纬度和年份信息
lon = data['lon'].values
lat = data['lat'].values
years = data['year'].values

# 创建空数组来存储每个经纬度格点的趋势
trend = np.zeros((len(lat), len(lon)))

# 逐个经纬度格点计算线性趋势
for i in range(len(lat)):
    for j in range(len(lon)):
        # 提取每个经纬度格点在各年的 PV 数据
        pv_series = pv_data[:, :, i, j].mean(dim='month')  # 对月份求平均值，得到逐年变化的 PV 值
        # 如果 PV 数据中存在 NaN，跳过该点的趋势计算
        if np.all(np.isnan(pv_series)):
            trend[i, j] = np.nan
        else:
            # 使用 np.polyfit 计算线性趋势，度数为1
            trend[i, j] = np.polyfit(years - years[0], pv_series, 1)[0]  # 返回斜率（趋势）

# 设置颜色映射范围（更密集的级别）
levels = np.linspace(-0.005, 0.005, 51)  # 增加颜色变化的分级密度
cmap = plt.get_cmap('RdBu_r')  # 使用对比强烈的红蓝色图
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# 绘制 PVpot 变化趋势的地区分布图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())  # 设置地图范围，覆盖中国全境

# 绘制趋势图，使用调整后的颜色映射
contour = ax.contourf(lon, lat, trend, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
cbar = plt.colorbar(contour, ax=ax, orientation='vertical', label='PV_POT Trend (decade^-1)')

# 添加国界线和海岸线
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE)

# 设置特定的经纬度刻度位置
ax.set_xticks(np.arange(70, 145, 10), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(15, 60, 10), crs=ccrs.PlateCarree())

# 南海小图
ax_inset = fig.add_axes([0.57, 0.2, 0.2, 0.2], projection=ccrs.PlateCarree())
ax_inset.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())  # 设置南海区域的范围

# 在南海区域绘制趋势图
ax_inset.contourf(lon, lat, trend, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

# 添加南海的国界线和海岸线
ax_inset.add_feature(cfeature.BORDERS, linestyle=':')
ax_inset.add_feature(cfeature.COASTLINE)

# 设置标题和标签
ax.set_title('Trend of PVpot in China', fontsize=14)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# 调整布局避免重叠
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

plt.show()
