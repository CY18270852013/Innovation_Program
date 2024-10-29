# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 23:06:54 2024

@author: Chen Yong
"""

import numpy as np
import matplotlib.pyplot as plt

# 定义文件路径
temperature_file = r'C:\Users\Chen Yong\Desktop\1951.1-2021.7\temperature\t1601.txt'
precipitation_file = r'C:\Users\Chen Yong\Desktop\1951.1-2021.7\precipitation\r1601.txt'

# 读取温度数据
with open(temperature_file, 'r') as f:
    temp_data = f.read().split()
temp_data = np.array(temp_data, dtype=float)
# 转换为℃单位
temp_data = temp_data * 0.1

# 读取降水量数据
with open(precipitation_file, 'r') as f:
    precip_data = f.read().split()
precip_data = np.array(precip_data, dtype=float)
# 降水量单位为mm，无需转换

# 赣州站点编号（假设站点编号从1到160，对应数组索引0到159）
station_index = 75  # 站点76对应索引75

# 提取赣州站点的数据
# 每年有160个数据点，71年
years = np.arange(1951, 2022)
num_years = len(years)

# 检查数据长度是否符合预期
expected_length = num_years * 160
if temp_data.size < expected_length or precip_data.size < expected_length:
    raise ValueError("If the data length is insufficient, check whether the data file is complete.")

# 重塑数据为(year, station)格式
temp_data = temp_data[:expected_length].reshape((num_years, 160))
precip_data = precip_data[:expected_length].reshape((num_years, 160))

# 提取赣州站点的1月份数据
ganzhou_temp = temp_data[:, station_index]
ganzhou_precip = precip_data[:, station_index]

# 绘图
fig, ax1 = plt.subplots(figsize=(14, 7))

color_temp = 'tab:red'
ax1.set_xlabel('Years', fontsize=14)
ax1.set_ylabel('Average temperature in January(°C)', color=color_temp, fontsize=14)
ax1.plot(years, ganzhou_temp, color=color_temp, label='Average temperature in January')
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.grid(True)

ax2 = ax1.twinx()  # 创建第二个y轴
color_precip = 'tab:blue'
ax2.set_ylabel('Average precipitation in January(mm)', color=color_precip, fontsize=14)
ax2.plot(years, ganzhou_precip, color=color_precip, label='Average precipitation in January')
ax2.tick_params(axis='y', labelcolor=color_precip)

# 添加图标题
plt.title('Interannual variation of mean temperature and precipitation at Ganzhou site from 1951 to January 2021', fontsize=16)

# 添加图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.show()

# 计算1951-1980年和1981-2010年的气候平均值
# 定义两个时期
period1_years = (years >= 1951) & (years <= 1980)
period2_years = (years >= 1981) & (years <= 2010)

# 计算平均值
temp_period1_avg = np.mean(ganzhou_temp[period1_years])
temp_period2_avg = np.mean(ganzhou_temp[period2_years])

precip_period1_avg = np.mean(ganzhou_precip[period1_years])
precip_period2_avg = np.mean(ganzhou_precip[period2_years])

# 打印结果
print("1951-1980年赣州站点1月平均气温: {:.2f} °C".format(temp_period1_avg))
print("1951-1980年赣州站点1月平均降水量: {:.2f} mm".format(precip_period1_avg))
print("1981-2010年赣州站点1月平均气温: {:.2f} °C".format(temp_period2_avg))
print("1981-2010年赣州站点1月平均降水量: {:.2f} mm".format(precip_period2_avg))
