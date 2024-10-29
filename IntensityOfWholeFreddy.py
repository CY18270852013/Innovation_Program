import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date

# 打开数据集
nc_file = r'C:\Users\Chen Yong\Desktop\Temporary desktop file\Research and cultivation project\IBTrACS.SI.v04r00.nc'
dataset = Dataset(nc_file)

# 读取需要的变量
dates = dataset.variables['time']
pressure = dataset.variables['usa_pres']
wind_speed = dataset.variables['usa_wind']
sid = dataset.variables['usa_atcf_id']

# 转换时间格式
units = dates.units
calendar = dates.calendar
dates = num2date(dates[:], units=units, calendar=calendar)

# 确定时间范围
start_date = np.datetime64('2023-01-01')
end_date = np.datetime64('2023-12-31')
start_index = np.where(dates == start_date)[0][0]
end_index = np.where(dates == end_date)[0][0]

# 筛选特定SID的数据
selected_sid = '2023061S22036'
sid_indices = np.where(sid[:] == selected_sid)[0]

# 选择时间范围内的数据
selected_dates = dates[start_index:end_index + 1]
selected_pressure = pressure[sid_indices, start_index:end_index + 1]
selected_wind_speed = wind_speed[sid_indices, start_index:end_index + 1]

# 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(selected_dates, selected_pressure[0], label='Central Pressure')
plt.plot(selected_dates, selected_wind_speed[0], label='Max Wind Speed')
plt.xlabel('Date')
plt.ylabel('Intensity')
plt.title(f'Tropical Cyclone Intensity Variation ({selected_sid})')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()