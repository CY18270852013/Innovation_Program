import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# 文件路径
file_path = r'C:\Users\Chen Yong\Desktop\InnovationProgram\ERA5 monthly averaged data on pressure levels of 2023.nc'

# 读取NetCDF文件
data = xr.open_dataset(file_path)

# 打印数据集的变量信息（可选）
#print(data)

# 提取U风分量和V风分量
u_wind = data['u'].isel(pressure_level=0)  # 选择100 hPa层
v_wind = data['v'].isel(pressure_level=0)  # 选择100 hPa层

# 创建地图投影
proj = ccrs.PlateCarree()

# 设置图形和子图
fig, axes = plt.subplots(3, 4, figsize=(20, 15), subplot_kw={'projection': proj})

# 设置颜色映射
cmap = plt.get_cmap('coolwarm')

# 逐月绘制数据
for i, ax in enumerate(axes.flat):
    # 提取每个月的数据（2D数据）
    u_month = u_wind.isel(date=i)  # 选择第i个月的U风分量

    # 绘制风速的颜色映射，西风（正值）为暖色，东风（负值）为冷色
    contour = ax.contourf(u_month.longitude, u_month.latitude, u_month, 
                          cmap=cmap, levels=np.linspace(-20, 20, 21), extend='both')
    
    # 设置地图特征
    ax.coastlines()
    ax.set_title(f'Month {i + 1}', fontsize=16, loc='center', pad=10)  # 减小标题的上边距

# 添加颜色条，放置在图的最下方
cbar = fig.colorbar(contour, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.02, aspect=50)
cbar.set_label('U Component of Wind at 100 hPa (m/s)', fontsize=16)
cbar.ax.tick_params(labelsize=13)  # 调整颜色条的刻度标签大小

# 添加总标题
plt.suptitle('Monthly Variations of U Component of Wind at 100 hPa (2023)', fontsize=22, y=0.88)  # 调整y值以放低标题

plt.tight_layout(pad=1.0, rect=[0, 0, 0, 0.95])  # 设置整体间距和留出空间给总标题
plt.show()
