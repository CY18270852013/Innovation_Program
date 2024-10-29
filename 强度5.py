import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# 打开IBTrACS数据文件
filename = r'C:\Users\Chen Yong\Desktop\Temporary desktop file\Research and cultivation project\IBTrACS.SI.v04r00.nc'
data = nc.Dataset(filename)

# 获取数据变量
pres = data.variables['usa_pres']
wind = data.variables['usa_wind']
atcf_id = data.variables['usa_atcf_id']
time = data.variables['time']

# 获取时间、最大风速和中心气压数据
dates = nc.num2date(time[:], time.units)
max_wind_speed = wind[:]
center_pressure = pres[:]

# 获取热带气旋"FREDDY"的索引
freddy_index = np.where(atcf_id[:] == b'SI112023')[0]

# 获取热带气旋"FREDDY"的最大风速和中心气压
freddy_max_wind_speed = max_wind_speed[freddy_index, :]
freddy_center_pressure = center_pressure[freddy_index, :]

# 绘制最大风速随时间变化的图表
plt.figure(figsize=(10, 6))
plt.plot(dates, freddy_max_wind_speed.squeeze(), marker='o')
plt.xlabel('Date')
plt.ylabel('Max Wind Speed (kts)')
plt.title('Max Wind Speed of Tropical Cyclone "FREDDY" in the Indian Ocean (2023)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制中心气压随时间变化的图表
plt.figure(figsize=(10, 6))
plt.plot(dates, freddy_center_pressure.squeeze(), marker='o')
plt.xlabel('Date')
plt.ylabel('Center Pressure (hPa)')
plt.title('Center Pressure of Tropical Cyclone "FREDDY" in the Indian Ocean (2023)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 关闭数据文件
data.close()