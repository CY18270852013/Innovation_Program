# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 00:14:56 2024

@author: Lenovo
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 定义文件路径
file_path = r'C:\Users\Chen Yong\Desktop\InnovationProgram\PV_2007_2016.nc'

# 加载数据
data = xr.open_dataset(file_path)

# 提取AOD和PV数据
aod = data['AOD']
pv = data['PV']

# 选择华北地区的经纬度范围
# 假设华北的定义为经度110°E到120°E，纬度32°N到42°N
lon_range = slice(110, 120)
lat_range = slice(32, 42)

# 选择6-8月的数据
months = [6,7,8]

# 选择指定年份、月份和地区的AOD和PV数据
selected_aod = aod.sel(year=slice(2007, 2022), month=months, lon=lon_range, lat=lat_range)
selected_pv = pv.sel(year=slice(2007, 2022), month=months, lon=lon_range, lat=lat_range)

# 将填充值设置为NaN
selected_aod = xr.where(selected_aod != -9999.0, selected_aod, np.nan)
selected_pv = xr.where(selected_pv != -9999.0, selected_pv, np.nan)

# 将数据转换为数组
aod_data = selected_aod.values.flatten()
pv_data = selected_pv.values.flatten()

# 移除NaN值
valid_indices = ~np.isnan(aod_data) & ~np.isnan(pv_data)
aod_data = aod_data[valid_indices]
pv_data = pv_data[valid_indices]

# 计算相关系数和显著性p值
corr, p_value = pearsonr(aod_data, pv_data)

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(aod_data, pv_data, alpha=0.5, s=8)  # 减小点的直径
plt.title(f'Relationship between PVpot and AOD in North China in summer\nCorrelation: {corr:.2f}',fontsize=16)
plt.xlabel('AOD', fontsize=18)  # 设置x轴标签字体大小
plt.ylabel('PV$_{POT}$', fontsize=18)  # 设置y轴标签字体大小

# 设置刻度字体大小
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 计算线性回归线
slope, intercept = np.polyfit(aod_data, pv_data, 1)
line = slope * aod_data + intercept
plt.plot(aod_data, line, color='red', label=f'Linear fit: y={slope:.2f}x+{intercept:.2f}')

# 显示显著性检验结果
if p_value < 0.01:
    plt.text(0.80, 0.13, 'p<0.01', transform=plt.gca().transAxes, fontsize=14, bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.5))

plt.legend(fontsize=16)
plt.grid(True)
plt.show()