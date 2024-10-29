import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt

# 读取netCDF文件
file_path = r'C:\Users\Chen Yong\Desktop\InnovationProgram\PV_2007_2016.nc'
data = xr.open_dataset(file_path)

# 提取需要的变量
I = data['DSW']  # 短波辐射
T = data['Tas']  # 气温
W = data['Wind']  # 风速
PV = data['PV']   # 光伏发电量
years = data['year'].values

# 定义性能比参数
Istc = 1000  # 标准测试条件下的短波辐射 W/m^2
Tstc = 25    # 标准测试条件下的温度，单位：℃ 
Y = 0.005    # 性能比温度系数，单晶硅太阳能电池

# 计算电池温度 Tcell
c1, c2, c3, c4 = 2, 0.95, 0.05, 1.5  # 假设系数
Tcell = c1 + c2 * T + c3 * I - c4 * W

# 计算性能比 PR
PR = 1 - Y * (Tcell - Tstc)

# 计算PVpot
PVpot = PR * I / Istc

# 定义不同地区的经纬度范围
regions = {
    "North China": [(34, 42), (114, 120)],
    "Northeast China": [(42, 54), (122, 135)],
    "East China": [(25, 34), (118, 123)],
    "Central China": [(28, 34), (108, 118)],
    "South China": [(20, 28), (110, 117)],
    "Southwest China": [(22, 34), (98, 108)],
    "Northwest China": [(34, 42), (85, 110)],
}

# 计算每个地区的平均PVpot值
region_avg_pvpot = {}
for region, ((lat_min, lat_max), (lon_min, lon_max)) in regions.items():
    region_data = PVpot.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    region_avg_pvpot[region] = region_data.mean(dim=('lat', 'lon'))

# 定义不同地区的折线符号和颜色
markers = ['o', 's', 'D', '^', 'v', 'P', '*']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# 绘制不同地区的PVpot年际变化折线图
plt.figure(figsize=(10, 6))

for i, (region, pvpot) in enumerate(region_avg_pvpot.items()):
    pvpot_yearly = pvpot.mean(dim='month')  # 对每年取平均
    plt.plot(years, pvpot_yearly, label=region, marker=markers[i], color=colors[i])

# 扩展y轴范围以避免图例覆盖折线
#plt.ylim([None, plt.ylim()[1] * 1.1])  # 将y轴最大值扩展10%

plt.xlabel('Year')
plt.ylabel('PVpot (W/m^2)')
plt.title('Interannual variation of PV potential in different regions of China (2007-2022)')

# 设置图例分成两行并放在上方
plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))

plt.grid(True)
plt.tight_layout()
plt.show()
