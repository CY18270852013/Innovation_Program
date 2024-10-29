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

# 对纬度、经度和月份维度进行平均，获得每年的 AOD 和 PV 数据
aod_annual = aod_data.mean(dim=['month', 'lat', 'lon'])
pv_annual = pv_data.mean(dim=['month', 'lat', 'lon'])

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
line1, = ax1.plot(years, aod_values, 'o-', color=color, label='AOD')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0.28, 0.38])  # 设置 AOD 的 y 轴范围

# 创建第二个 y 轴，用于绘制 PV
ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('PV$_{POT}$', color=color)
line2, = ax2.plot(years, pv_values, 'o-', color=color, label='PV$_{POT}$')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0.19, 0.20])  # 设置 PV 的 y 轴范围

# 添加图例，并将图例分为两行显示
lines = [line1, line2]
labels = [line1.get_label(), line2.get_label()]
# 使用 fig.legend 将图例放在 upper center，分成两行显示
fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

# 在图形的 upper center 添加相关系数的注释，作为图例的一部分
ax1.text(0.5, 1.05, corr_label, ha='center', va='center', transform=ax1.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

# 调整布局
fig.tight_layout()

# 显示图形
plt.show()
