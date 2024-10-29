# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:16:20 2024

@author: Chen Yong
"""

import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 读取 NetCDF 数据文件
file_path = r'C:\Users\Chen Yong\Desktop\InnovationProgram\PV_2007_2016.nc'
data = xr.open_dataset(file_path)

# 提取所需数据
years = data['year'].values
aod_data = data['AOD']  # 气溶胶光学厚度 (AOD)
I = data['DSW']  # 短波辐射 (DSW)
T = data['Tas']  # 气温 (Tas)
WS = data['Wind']  # 风速 (Wind)

# 计算 Tcell 和 PVpot 的步骤
c1, c2, c3, c4 = 2.56, 0.47, 0.065, 1.02  # 假设的常数值
Tstc = 25  # 标准测试条件下温度 (25°C)
Y = 0.005  # 单晶硅太阳能电池的Y系数
Istc = 1000  # 标准短波辐射 (1000 W/m²)

# 计算 Tcell
Tcell = c1 + c2 * T + c3 * I - c4 * WS

# 计算性能比 PR
PR = 1 - Y * (Tcell - Tstc)

# 计算光伏潜力 PVpot
PVpot = PR * I / Istc

# 对纬度、经度和月份维度进行平均，获得每年的 AOD 和 PVpot 数据
aod_annual = aod_data.mean(dim=['lat', 'lon', 'month'])
pvpot_annual = PVpot.mean(dim=['lat', 'lon','month'])

# 将数据转换为 1D NumPy 数组，传递给 pearsonr 函数
aod_values = aod_annual.values
pvpot_values = pvpot_annual.values

# 计算 AOD 和 PVpot 之间的相关系数
r, p_value = pearsonr(aod_values, pvpot_values)
corr_label = f'r = {r:.4f}, p < 0.01'  # 创建相关系数的标签

# 创建图形和双y轴
fig, ax1 = plt.subplots(figsize=(10, 5))

# 绘制 AOD 的折线图
color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('AOD', color=color)
line1, = ax1.plot(years, aod_values, 'o-', color='lightblue', label='AOD', linewidth=1.5)
ax1.tick_params(axis='y', labelcolor=color)

# 自动调整 AOD 的 y 轴范围，使得数据更加协调
AOD_min, AOD_max = aod_values.min(), aod_values.max()
AOD_margin = (AOD_max - AOD_min) * 0.1  # 给上下留出10%的余量
ax1.set_ylim([AOD_min - AOD_margin, AOD_max + AOD_margin])

# 创建第二个 y 轴，用于绘制 PVpot
ax2 = ax1.twinx()
ax2.set_ylabel('PV$_{POT}$', color='tab:orange')
line2, = ax2.plot(years, pvpot_values, 'o-', color='peru', label='PV$_{POT}$', linewidth=1.5)
ax2.tick_params(axis='y', labelcolor='tab:orange')

# 自动调整 PVpot 的 y 轴范围，使得数据更加协调
PV_min, PV_max = pvpot_values.min(), pvpot_values.max()
PV_margin = (PV_max - PV_min) * 0.1  # 给上下留出10%的余量
ax2.set_ylim([PV_min - PV_margin, PV_max + PV_margin])

# 添加图例，并将图例分为两行显示
lines = [line1, line2]
labels = [line1.get_label(), line2.get_label()]
fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.77, 1.0), ncol=2, bbox_transform=ax1.transAxes)

# 在图形的 upper center 添加相关系数的注释
ax1.text(0.8, 0.8, corr_label, ha='center', va='center', transform=ax1.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

# 调整布局
fig.tight_layout()

# 显示图形
plt.title('AOD & PVpot Yearly Variations in China')
plt.show()
