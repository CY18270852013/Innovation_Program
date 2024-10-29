import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

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

# 选择时间段 2023-09-27 到 2023-10-03
start_time = '2024-09-27'
end_time = '2024-10-03'
times = slice(start_time, end_time)

# 创建绘图
fig, axes = plt.subplots(2, 1, figsize=(14, 12), subplot_kw={'projection': ccrs.PlateCarree()})

# 地面图（温度、海平面气压和风场）
ax1 = axes[0]
temperature_2m.sel(valid_time=times).mean(dim='valid_time').plot(ax=ax1, cmap='coolwarm', cbar_kwargs={'label': '2m Temperature (°C)'})
ax1.quiver(u_wind.longitude, u_wind.latitude,
           u_wind.sel(valid_time=times).mean(dim='valid_time'),
           v_wind.sel(valid_time=times).mean(dim='valid_time'),
           color='black', scale=700)
msl_pressure.sel(valid_time=times).mean(dim='valid_time').plot.contour(ax=ax1, colors='black', levels=np.arange(990, 1030, 2))
ax1.set_title(f'Surface Temperature, Winds, and Sea Level Pressure\nAveraged from {start_time} to {end_time}', fontsize=16)
ax1.coastlines()

# 高空图（850 hPa 温度和等温线）
ax2 = axes[1]
temperature_850hPa.sel(valid_time=times).mean(dim='valid_time').plot(ax=ax2, cmap='coolwarm', cbar_kwargs={'label': '850 hPa Temperature (°C)'})
ax2.contour(temperature_850hPa.longitude, temperature_850hPa.latitude,
            temperature_850hPa.sel(valid_time=times).mean(dim='valid_time'),
            levels=np.arange(-15, 35, 5), colors='black')
ax2.set_title(f'850 hPa Temperature and Isotherms\nAveraged from {start_time} to {end_time}', fontsize=16)
ax2.coastlines()

# 调整布局
plt.tight_layout()
plt.show()
