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

# 选择中国地区的经纬度范围
lon_min, lon_max = 73.5, 135.0  # 中国的经度范围
lat_min, lat_max = 18.0, 54.0   # 中国的纬度范围

# 使用条件选择数据
aod_data_china = aod_data.where((data['lon'] >= lon_min) & (data['lon'] <= lon_max) &
                                  (data['lat'] >= lat_min) & (data['lat'] <= lat_max), drop=True)
pv_data_china = pv_data.where((data['lon'] >= lon_min) & (data['lon'] <= lon_max) &
                               (data['lat'] >= lat_min) & (data['lat'] <= lat_max), drop=True)

# 过滤时间范围为 2007 到 2022 年
aod_data_china = aod_data_china.sel(year=slice(2007, 2022))
pv_data_china = pv_data_china.sel(year=slice(2007, 2022))

# 对纬度、经度和月份维度进行平均，获得每年的 AOD 和 PV 数据
aod_annual = aod_data_china.mean(dim=['month', 'lat', 'lon'])
pv_annual = pv_data_china.mean(dim=['month', 'lat', 'lon'])

# 将数据转换为 1D NumPy 数组
aod_values = aod_annual.values
pv_values = pv_annual.values

# 计算 AOD 和 PV 之间的相关系数
r, p_value = pearsonr(aod_values, pv_values)
corr_label = f'r = {r:.4f}, p < 0.01'  # 创建相关系数的标签

# 创建图形和双y轴
fig, ax1 = plt.subplots(figsize=(10, 5))

# 绘制 AOD 的折线图
color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('AOD', color=color)
line1, = ax1.plot(years[years >= 2007], aod_values, 'o-', color=color, label='AOD')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0.28, 0.4])  # 设置 AOD 的 y 轴范围

# 创建第二个 y 轴，用于绘制 PV
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('PV$_{POT}$ (MW/m²)', color=color)  # 修改这里以包含单位
line2, = ax2.plot(years[years >= 2007], pv_values, 'o-', color=color, label='PV$_{POT}$')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.19, 0.20])  # 设置 PV 的 y 轴范围

# 添加图例，并将图例分为两行显示
lines = [line1, line2]
labels = [line1.get_label(), line2.get_label()]
# 使用 fig.legend 将图例放在 upper center，分成两行显示
fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.7, 0.9), ncol=2)

# 在图形的 upper center 添加相关系数的注释，作为图例的一部分
ax1.text(0.79, 0.75, corr_label, ha='center', va='center', transform=ax1.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

# 调整布局
fig.tight_layout()

# 显示图形
plt.title('AOD & PVpot Yearly Variations in China')
plt.show()
