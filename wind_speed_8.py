import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 加载ERA5数据
data = xr.open_dataset(r'C:\Users\Chen Yong\Downloads\PV_2007_2016.nc')

# 提取风速数据
wind_speed = data['Wind']

# 提取年份数据
years = data['year'].values.astype(int)

# 提取中国区域的经度和纬度范围
lon_range = slice(70, 140)
lat_range = slice(10, 60)

# 创建地图
fig = plt.figure(figsize=(16, 12))

# 定义季节-月份映射关系
seasons = {
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11],
    'Winter': [12, 1, 2]
}

# 绘制四季风速平均空间分布图
for i, (season, months) in enumerate(seasons.items()):
    # 提取指定季节的风速数据
    selected_data = wind_speed.sel(year=years[(years >= 2007) & (years <= 2022)],
                                   month=np.isin(data['month'].values, months),
                                   lon=lon_range, lat=lat_range)
    # 计算季节平均风速
    average_wind_speed = selected_data.mean(dim=['year', 'month'])
    
    # 提取经度、纬度和风速数据
    lon = average_wind_speed['lon'].values
    lat = average_wind_speed['lat'].values
    wind = average_wind_speed.values
    
    # 将风速数据中的填充值设置为NaN
    wind = np.where(wind == -9999.0, np.nan, wind)
    
    # 绘制子图
    ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
    
    # 绘制中国边境线
    ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)
    
    # 绘制地图边界、海岸线和陆地
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, color='lightgray')
    
    # 绘制经纬度网格线
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    
    # 绘制空间分布图
    im = ax.pcolormesh(lon, lat, wind, cmap='jet', transform=ccrs.PlateCarree(), antialiased=True)
    plt.colorbar(im, label='Wind Speed (m/s)', ax=ax)
    ax.set_title(f'Average Wind Speed ({season} {2007}-{2022})')

# 添加中国南海小图
south_china_sea_extent = [105, 125, 0, 25]
ax_inset = fig.add_axes([0.58, 0.2, 0.2, 0.2], projection=ccrs.PlateCarree())
ax_inset.coastlines(resolution='10m')
ax_inset.add_feature(cfeature.LAND, color='lightgray')
ax_inset.add_feature(cfeature.BORDERS, linestyle='-', alpha=0.5)

# 绘制中国领海边境线
china_boundaries = cfeature.NaturalEarthFeature(category='cultural',
                                               name='admin_0_boundary_lines_land',
                                               scale='10m',
                                               facecolor='none')
ax_inset.add_feature(china_boundaries, linestyle='-', edgecolor='black')

ax_inset.set_extent(south_china_sea_extent)

# 显示图形
plt.tight_layout()
plt.show()