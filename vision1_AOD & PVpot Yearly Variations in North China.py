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
pv_data = data['PV']    # 光伏发电潜力 (PV)

# 将填充值设置为NaN
aod_data = xr.where(aod_data != -9999.0, aod_data, np.nan)
pv_data = xr.where(pv_data != -9999.0, pv_data, np.nan)

# 设置华北地区的经纬度范围
lon_min, lon_max = 110.0, 120.0  # 华北地区的经度范围
lat_min, lat_max = 32.0, 42.0    # 华北地区的纬度范围

# 使用条件选择华北地区的数据
aod_data_north_china = aod_data.where((data['lon'] >= lon_min) & (data['lon'] <= lon_max) & 
                                      (data['lat'] >= lat_min) & (data['lat'] <= lat_max), drop=True)
pv_data_north_china = pv_data.where((data['lon'] >= lon_min) & (data['lon'] <= lon_max) & 
                                    (data['lat'] >= lat_min) & (data['lat'] <= lat_max), drop=True)

# 过滤时间范围为 2007 到 2022 年
aod_data_north_china = aod_data_north_china.sel(year=slice(2007, 2022))
pv_data_north_china = pv_data_north_china.sel(year=slice(2007, 2022))

# 对纬度、经度和月份维度进行平均，获得每年的 AOD 和 PV 数据
aod_annual = aod_data_north_china.mean(dim=['month', 'lat', 'lon'])
pv_annual = pv_data_north_china.mean(dim=['month', 'lat', 'lon'])

# 将数据转换为 1D NumPy 数组
aod_values = aod_annual.values
pv_values = pv_annual.values
years_filtered = years[(years >= 2007) & (years <= 2022)]  # 过滤后的年份

# 计算 AOD 和 PV 之间的相关系数
r, p_value = pearsonr(aod_values, pv_values)
corr_label = f'r = {r:.4f}, p < 0.01'  # 创建相关系数的标签

# 创建图形和双y轴
fig, ax1 = plt.subplots(figsize=(10, 5))

# 绘制 AOD 的折线图
color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('AOD', color=color)
line1, = ax1.plot(years_filtered, aod_values, 'o-', color=color, label='AOD')
ax1.tick_params(axis='y', labelcolor=color)

# 自动调整 AOD 的 y 轴范围，使得数据更加协调
AOD_min, AOD_max = aod_values.min(), aod_values.max()
AOD_margin = (AOD_max - AOD_min) * 0.1  # 给上下留出10%的余量
ax1.set_ylim([AOD_min - AOD_margin, AOD_max + AOD_margin])

# 创建第二个 y 轴，用于绘制 PV
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('PV$_{POT}$ (MW/m²)', color=color)
line2, = ax2.plot(years_filtered, pv_values, 'o-', color=color, label='PV$_{POT}$')
ax2.tick_params(axis='y', labelcolor=color)

# 自动调整 PV 的 y 轴范围
PV_min, PV_max = pv_values.min(), pv_values.max()
PV_margin = (PV_max - PV_min) * 0.1  # 给上下留出10%的余量
ax2.set_ylim([PV_min - PV_margin, PV_max + PV_margin])

# 添加图例，并将图例分为两行显示
lines = [line1, line2]
labels = [line1.get_label(), line2.get_label()]
fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.71, 0.9), ncol=2)

# 在图形的 upper center 添加相关系数的注释，作为图例的一部分
ax1.text(0.8, 0.77, corr_label, ha='center', va='center', transform=ax1.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

# 调整布局
fig.tight_layout()

# 显示图形
plt.title('AOD & PVpot Yearly Variations in North China')
plt.show()
