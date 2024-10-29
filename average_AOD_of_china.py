# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:39:27 2024

@author: Chen Yong
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 加载ERA5数据
data = xr.open_dataset(r'C:\Users\Chen Yong\Downloads\PV_2007_2016.nc')

# 提取AOD数据
aod = data["AOD"]

# 提取年份数据
years = data["year"].values.astype(int)
months = data["month"].values.astype(int)

# 提取中国区域的经度和纬度范围
lon_range = slice(70, 140)
lat_range = slice(0, 60)

# 提取指定年份和中国区域的AOD数据
selected_data = aod.sel(year=years, month=months, lat=lat_range, lon=lon_range)

# 计算平均AOD
mean_aod = selected_data.mean(dim=["year", "month"])

# 提取经度、纬度和平均AOD数据
lon = data["lon"].values
lat = data["lat"].values
aod_values = mean_aod.values
# 将AOD数据中的填充值设置为NaN
aod_values = np.where(aod_values == -9999.0, np.nan, aod_values)

# 创建地图
fig = plt.figure(figsize=(10, 8), dpi=1000, facecolor="white")  # 设置背景为白色

# 创建colorbar
cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])

# 绘制主地图
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# 绘制中国边境线
ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)

# 绘制地图边界、海岸线和陆地
ax.coastlines(resolution="10m")
ax.add_feature(cfeature.LAND, color="lightgray")

# 绘制经纬度网格线
ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)

# 将超出中国边境范围的AOD值设置为NaN
china_land_aod = mean_aod.sel(lat=lat_range, lon=lon_range)
china_land_aod_values = china_land_aod.values
china_land_aod_values = np.where(china_land_aod_values == -9999.0, np.nan, china_land_aod_values)

# 绘制空间分布图（使用掩码处理）
masked_aod_values = np.ma.masked_invalid(aod_values)
im = ax.pcolormesh(lon, lat, masked_aod_values, cmap="jet", transform=ccrs.PlateCarree(), shading='auto')
plt.colorbar(im, label="AOD", cax=cax)
plt.title("AOD annual mean in China (2007-2022)")

# 添加中国南海小图
south_china_sea_extent = [105, 125, 0, 25]
ax_inset = fig.add_axes([0.7, 0.2, 0.2, 0.2], projection=ccrs.PlateCarree())
ax_inset.coastlines(resolution="10m")
ax_inset.add_feature(cfeature.LAND, color="lightgray")
ax_inset.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.5)

# 绘制中国领海边境线
china_boundaries = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_0_boundary_lines_land",
    scale="10m",
    facecolor="none",
)
ax_inset.add_feature(china_boundaries, linestyle="-", edgecolor="black")

ax_inset.set_extent(south_china_sea_extent)

# 计算中国陆地部分的AOD多年平均分布
china_land_aod = mean_aod.sel(lat=lat_range, lon=lon_range)
china_land_aod_values = china_land_aod.values
china_land_aod_values = np.where(china_land_aod_values == -9999.0, np.nan, china_land_aod_values)# 将超出中国边境范围的AOD值设置为NaN


# 绘制中国陆地的AOD多年平均分布（使用掩码处理）
masked_china_land_aod_values = np.ma.masked_invalid(china_land_aod_values)
ax_inset.imshow(
    masked_china_land_aod_values,
    cmap="jet",
    origin="lower",
    extent=[lon_range.start, lon_range.stop, lat_range.start, lat_range.stop],
    transform=ccrs.PlateCarree(),
    alpha=0.7,
)

# 显示图形
plt.show()