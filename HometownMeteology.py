# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 22:32:40 2024

@author: Chen Yong
"""

import numpy as np
import matplotlib.pyplot as plt

# 读取数据函数
def read_data(filename):
    data = np.loadtxt(filename, skiprows=1)
    return data

# 主函数
def main():
    # 赣州站点数据路径
    temp_path = r'C:\Users\Chen Yong\Desktop\1951.1-2021.7\temperature\t1601.txt'
    precip_path = r'C:\Users\Chen Yong\Desktop\1951.1-2021.7\precipitation\r1601.txt'
    
    # 读取数据
    temperatures = read_data(temp_path)
    precipitations = read_data(precip_path)
    
    # 只取1月份的数据
    january_temps = temperatures[:, 0]  # 假设1月份数据在第一列
    january_precips = precipitations[:, 0]  # 假设1月份数据在第一列
    
    # 年份
    years = np.arange(1951, 2022)
    
    # 计算平均值
    avg_temp_1951_1980 = np.mean(january_temps[0:30])
    avg_precip_1951_1980 = np.mean(january_precips[0:30])
    avg_temp_1981_2010 = np.mean(january_temps[30:60])
    avg_precip_1981_2010 = np.mean(january_precips[30:60])
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(years, january_temps, label='Average Temperature (°C)', color='red')
    plt.plot(years, january_precips, label='Average Precipitation (mm)', color='blue')
    plt.title('January Average Temperature and Precipitation in Ganzhou (1951-2021)')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 输出平均值
    print("Average Temperature (1951-1980):", avg_temp_1951_1980)
    print("Average Precipitation (1951-1980):", avg_precip_1951_1980)
    print("Average Temperature (1981-2010):", avg_temp_1981_2010)
    print("Average Precipitation (1981-2010):", avg_precip_1981_2010)

if __name__ == "__main__":
    main()