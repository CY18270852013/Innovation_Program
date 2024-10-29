import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from shapely.geometry import box

# 设置字体大小
plt.rcParams["font.size"] = 13

# 读取数据
data = xr.open_dataset(r"C:\Users\Chen Yong\Desktop\InnovationProgram\PV_2007_2016.nc")

# 提取PV数据
pv = data['PV']

# 选择特定月份的数据
spring = [3, 4, 5]
summer = [6, 7, 8]
fall = [9, 10, 11]
winter = [12, 1, 2]
year = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# 获取实际的经度和纬度数组
lon = data['lon'].values
lat = data['lat'].values

# 选择指定年份、月份和地区的PV数据
spring_pv = pv.sel(year=slice(2007, 2022), month=spring)
summer_pv = pv.sel(year=slice(2007, 2022), month=summer)
fall_pv = pv.sel(year=slice(2007, 2022), month=fall)
winter_pv = pv.sel(year=slice(2007, 2022), month=winter)
year_pv = pv.sel(year=slice(2007, 2022), month=year)

# 将填充值设置为NaN
spring_pv = xr.where(spring_pv != -9999.0, spring_pv, np.nan)
summer_pv = xr.where(summer_pv != -9999.0, summer_pv, np.nan)
fall_pv = xr.where(fall_pv != -9999.0, fall_pv, np.nan)
winter_pv = xr.where(winter_pv != -9999.0, winter_pv, np.nan)
year_pv = xr.where(year_pv != -9999.0, year_pv, np.nan)

# 计算各季节和年平均的PV值
spring_mean = spring_pv.mean(dim=('year', 'month'))
summer_mean = summer_pv.mean(dim=('year', 'month'))
fall_mean = fall_pv.mean(dim=('year', 'month'))
winter_mean = winter_pv.mean(dim=('year', 'month'))
year_mean = year_pv.mean(dim=('year', 'month'))
# 创建自定义colormap
colors = ["blue", "cyan",  "yellow", "red"]  # 蓝色过渡到青色过渡到绿色过渡到黄色过渡到红色
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 定义绘图函数
def plot_seasonal_mean(mean_data, title, ax, vmin, vmax):
    mean_data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=True, cbar_kwargs={'ticks': [0, 0.1, 0.2, 0.3, 0.4]}, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
    ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', edgecolor='black', facecolor='none'))
    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())  # 设置中国区域范围

    # 添加南海小图
    ax_inset = ax.inset_axes([0.7, 0.1, 0.3, 0.3], projection=ccrs.PlateCarree())
    mean_data.plot(ax=ax_inset, transform=ccrs.PlateCarree(), cmap=cmap, add_colorbar=False, vmin=vmin, vmax=vmax)
    ax_inset.set_extent([105, 125, 0, 25], crs=ccrs.PlateCarree())
    ax_inset.add_feature(cfeature.COASTLINE)
    ax_inset.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
    ax_inset.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', edgecolor='black', facecolor='none'))

    # 设置南海小图的经纬度刻度
    ax_inset.set_xticks([106, 125], crs=ccrs.PlateCarree())
    ax_inset.set_yticks([0, 25], crs=ccrs.PlateCarree())
    ax_inset.tick_params(labelsize=10, top=False, right=False, labelleft=True, labelbottom=True)  # 隐藏右上侧标签
    ax_inset.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax_inset.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    ax_inset.tick_params(top=False, right=False)
        # 删除轴标签 "lon" 和 "lat"
    ax_inset.xaxis.label.set_visible(False)
    ax_inset.yaxis.label.set_visible(False)

    # 设置主图的经纬度刻度，只在左侧显示纬度，只在底部显示经度
    ax.set_xticks([72, 90, 108, 126], crs=ccrs.PlateCarree())
    ax.set_yticks([20, 30, 40, 50], crs=ccrs.PlateCarree())
    ax.tick_params(labelsize=13, top=False, right=False, labelleft=True, labelbottom=True)  # 隐藏右上侧标签
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

        # 删除轴标签 "lon" 和 "lat"
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

# 创建图形和子图
fig, axes = plt.subplots(2, 2, figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})

# 设置颜色条范围
vmin = 0
vmax = 0.4

# 绘制各季节平均图
plot_seasonal_mean(spring_mean, 'Spring Mean PV', axes[0, 0], vmin, vmax)
plot_seasonal_mean(summer_mean, 'Summer Mean PV', axes[0, 1], vmin, vmax)
plot_seasonal_mean(fall_mean, 'Fall Mean PV', axes[1, 0], vmin, vmax)
plot_seasonal_mean(winter_mean, 'Winter Mean PV', axes[1, 1], vmin, vmax)

# 调整布局
plt.tight_layout()

# 单独绘制年平均图
fig_year, ax_year = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
plot_seasonal_mean(year_mean, 'Yearly Mean PV', ax_year, vmin, vmax)

# 显示图形
plt.show()