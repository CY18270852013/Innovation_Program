# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:16:35 2024

@author: Lenovo
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cnmaps import get_adm_maps, draw_maps, clip_contours_by_map, draw_map

#设置字体大小
plt.rcParams["font.size"] = 13

#读取数据
data = xr.open_dataset(r'C:\Users\Chen Yong\Desktop\InnovationProgram\PV_2007_2016.nc')

#提取AOD和PV数据
aod = data['AOD']
pv = data['PV']

#选择6-8月的数据
months = [6, 7, 8] # 6, 7, 8三个月，每年重复三次

#获取实际的经度和纬度数组
lon = data['lon'].values
lat = data['lat'].values

#选择指定年份、月份和地区的AOD和PV数据
selected_aod = aod.sel(year=slice(2007, 2022), month=months)
selected_pv = pv.sel(year=slice(2007, 2022), month=months)

#将填充值设置为NaN
selected_aod = xr.where(selected_aod != -9999.0, selected_aod, np.nan)
selected_pv = xr.where(selected_pv != -9999.0, selected_pv, np.nan)

#计算相关系数
correlation_matrix = np.zeros((len(lat), len(lon)))
p_value_matrix = np.zeros((len(lat), len(lon)))

for i in range(len(lat)):
    for j in range(len(lon)):
        # 提取当前格点的时间序列
        aod_values = selected_aod.isel(lat=i, lon=j).values
        pv_values = selected_pv.isel(lat=i, lon=j).values
        
        # 移除NaN值
        valid = ~np.isnan(aod_values) & ~np.isnan(pv_values)
        if np.sum(valid) > 1:  # 确保有足够的数据点进行相关性分析
            corr, p_value = pearsonr(aod_values[valid], pv_values[valid])
            correlation_matrix[i, j] = corr
            p_value_matrix[i, j] = p_value
#创建自定义colormap
colors = ["blue", "white", "red"] # 负值蓝色，0值白色，正值红色
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

#定义画布
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 8))

#定义绘制大地图的函数
def map_plot(fig, ax, lat, lon, data, p_value_data, is_mask, is_province_boundary, title):
    big_map = get_adm_maps(country='中华人民共和国', level='国')
    big_map_oneline = get_adm_maps(country='中华人民共和国', level='国', record='first', only_polygon=True)
    cf = ax.contourf(lon, lat, data, cmap=cmap, transform=ccrs.PlateCarree(), levels=np.linspace(-1, 1, 256))
    if is_mask:
        clip_contours_by_map(cf, big_map_oneline)
    draw_maps(big_map, linewidth=1.2, color='k')
    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
    ax.set_title(title)
    
    gl = ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.1, x_inline=False, y_inline=False, color='k')
    gl.top_labels = False
    gl.right_labels = False
    gl.rotate_labels = None
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    cbar = fig.colorbar(cf, ax=ax, shrink=0.9, extendfrac='auto', extendrect=True, location='bottom', fraction=0.05, pad=0.08)
    cbar.set_label('Correlation Coefficient')
    cbar.set_ticks([0.8, 0.6, 0.4, 0.2, 0, -0.2, -0.4, -0.6, -0.8])

# 显著性检验区域加粗
    contour = ax.contour(lon, lat, p_value_data, levels=[0.01], colors='black', linewidths=1.5, linestyles='solid')
# 定义添加南海小地图的函数
def add_nanhai(ax, pos, lat, lon, data):
    ax_nanhai = fig.add_axes(pos, projection=ccrs.PlateCarree())
    lon1, lon2, lat1, lat2 = 105, 125, 0, 25
    box_nanhai = [lon1, lon2, lat1, lat2]
    ax_nanhai.set_extent(box_nanhai, crs=ccrs.PlateCarree())
    nanhai = get_adm_maps(country='中华人民共和国', level='国')
    draw_maps(nanhai, linewidth=0.8, color='k')
    cf_nanhai = ax_nanhai.contourf(lon, lat, data, cmap=cmap, transform=ccrs.PlateCarree(), levels=np.linspace(-1, 1, 256))
    ax_nanhai.text(-0.15, 0.05, '0°N', transform=ax_nanhai.transAxes, fontsize=10, ha='center', va='top')
    ax_nanhai.text(0.05, -0.19, '105°E', transform=ax_nanhai.transAxes, fontsize=10, ha='center', va='bottom')
    ax_nanhai.text(-0.05, 1.01, '25°N', transform=ax_nanhai.transAxes, fontsize=10, ha='right', va='center')
    ax_nanhai.text(0.84, -0.13, '125°E', transform=ax_nanhai.transAxes, fontsize=10, ha='left', va='center')

# 绘制大地图
ax1 = fig.add_subplot(1, 1, 1, projection=proj)
map_plot(fig, ax1, lat, lon, correlation_matrix, p_value_matrix, True, True, 'Correlation between AOD and PV (Jun-Aug)')

# 添加南海小地图
pos1 = [0.75, 0.25, 0.15, 0.15]  # 南海小地图位置和长宽,根据画布自己调试
add_nanhai(ax1, pos1, lat, lon, correlation_matrix)  # 添加南海小地图

plt.show()