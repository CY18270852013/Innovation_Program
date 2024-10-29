# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 01:36:04 2024

@author: Chen Yong
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd  # 添加这一行以导入pandas库

# 文件路径
surface_file = r'C:\Users\Chen Yong\Desktop\InnovationProgram\surface_data.nc'
high_altitude_file = r'C:\Users\Chen Yong\Desktop\InnovationProgram\high_altitude_data.nc'

# 读取地面数据
surface_data = xr.open_dataset(surface_file)
temperature_2m = surface_data['t2m'] - 273.15  # 转换为摄氏度
u_wind = surface_data['u10']
v_wind = surface_data['v10']
msl_pressure = surface_data['msl'] / 100  # 转换为 hPa

# 读取高空数据
high_altitude_data = xr.open_dataset(high_altitude_file)
temperature_850hPa = high_altitude_data['t'].isel(pressure_level=0) - 273.15  # 转换为摄氏度

# 选择时间段 2024-09-27 到 2024-10-03
start_time = '2024-09-27'
end_time = '2024-10-03'
times = pd.date_range(start=start_time, end=end_time, freq='D')  # 生成日期序列

# 创建图形
fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

# 循环生成每一天的数据
for i, time in enumerate(times):
    ax = axes[i]
    
    # 绘制地面温度
    temperature_2m.sel(valid_time=time).plot(ax=ax, cmap='coolwarm', add_colorbar=False)
    
    # 绘制地面风场
    ax.quiver(u_wind.longitude, u_wind.latitude,
              u_wind.sel(valid_time=time), v_wind.sel(valid_time=time),
              color='black', scale=700, label='Surface Winds')
    
    # 绘制地面海平面气压等压线
    msl_pressure.sel(valid_time=time).plot.contour(ax=ax, colors='blue', levels=np.arange(990, 1030, 2), label='MSL Pressure')
    
    # 绘制高空850 hPa等温线
    ax.contour(temperature_850hPa.longitude, temperature_850hPa.latitude,
               temperature_850hPa.sel(valid_time=time), levels=np.arange(-15, 35, 5),
               colors='red', label='850 hPa Isotherms')
    
    # 设置标题
    ax.set_title(f'Cold Air Process on {time.strftime("%Y-%m-%d")}', fontsize=14)
    ax.coastlines()

# 添加整体标题
fig.suptitle('Cold Air Activity Process (Surface Fronts and 850 hPa Isotherms) From 2024-09-27 to 2024-10-03', fontsize=20, x = 0.45, y = 0.85)

# 添加图例说明和注释信息
fig.text(0.89, 0.2, 'Note: \nBlack arrows represent surface winds.\nBlue contours indicate sea level pressure.\nRed contours represent 850 hPa isotherms.',
         fontsize=14, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5))

# 调整布局
plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 调整图形布局以留出空间
plt.show()
