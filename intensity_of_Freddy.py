import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date

# 打开数据集
nc_file = r'C:\Users\Chen Yong\Desktop\Temporary desktop file\Research and cultivation project\IBTrACS.SI.v04r00.nc'
dataset = Dataset(nc_file)

# 读取需要的变量
dates = dataset.variables['time']
pressure = dataset.variables['central_pressure']
wind_speed = dataset.variables['max_wind_speed']

# 转换时间格式
units = dates.units
calendar = dates.calendar
dates = num2date(dates[:], units=units, calendar=calendar)

# 确定时间范围
start_date = np.datetime64('2023-02-03')
end_date = np.datetime64('2023-03-14')
start_index = np.where(dates == start_date)[0][0]
end_index = np.where(dates == end_date)[0][0]

# 选择时间范围内的数据
selected_dates = dates[start_index:end_index + 1]
selected_pressure = pressure[start_index:end_index + 1]
selected_wind_speed = wind_speed[start_index:end_index + 1]

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(selected_dates, selected_pressure, label='Central Pressure')
plt.plot(selected_dates, selected_wind_speed, label='Max Wind Speed')
plt.xlabel('Date')
plt.ylabel('Intensity')
plt.title('Tropical Cyclone Intensity Variation (South Indian Ocean)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()