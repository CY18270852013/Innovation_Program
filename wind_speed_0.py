import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# 读取ERA5的nc数据
data = xr.open_dataset(r'C:\Users\Chen Yong\Downloads\PV_2007_2016.nc')

# 提取中国范围的数据，假设数据中经度(lon)从0到360，纬度(lat)从-90到90
china_data = data.sel(lon=slice(70, 140), lat=slice(0, 65))

# 计算2007年到2016年中国的风速平均值（注意文件名称表明数据仅到2016年）
china_wind_speed = china_data['Wind'].sel(year=slice('2007-01-01', '2016-12-31')).mean(dim='year')

# 确保结果是2D
china_wind_speed = china_wind_speed.squeeze()  # 将长度为1的维度去除

# 绘制空间分布图
plt.figure(figsize=(10, 6))
plt.contourf(china_wind_speed.lon, china_wind_speed.lat, china_wind_speed, cmap='coolwarm', levels=np.linspace(china_wind_speed.min(), china_wind_speed.max(), 100))
plt.colorbar(label='Wind Speed (m/s)')
plt.title('Average Wind Speed in China (2007-2016)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()
