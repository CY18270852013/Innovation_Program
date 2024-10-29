import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取数据
ds = xr.open_dataset(r'C:\Users\Chen Yong\Desktop\Temporary desktop file\Research and cultivation project\olr-daily_v01r02-preliminary_20230101_20231219.nc')

# 选择所需的区域和时间范围
ds_subset = ds.sel(lat=slice(-10, -25), lon=slice(30, 120), time=slice('2023-02-06', '2023-03-14'))

# 计算纬向平均
olr_mean = ds_subset.olr.mean(dim='lat')

# 绘制 Hovmuller 图
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.pcolormesh(ds_subset.lon, ds_subset.time, olr_mean, cmap='RdBu_r', vmin=-30, vmax=30)
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Date')
ax.set_title('Outgoing Longwave Radiation (OLR) Hovmuller Diagram')
ax.set_xticks(np.arange(30, 121, 15))
ax.set_xticklabels([str(int(x)) for x in np.arange(30, 121, 15)])
ax.set_yticks(ds_subset.time[::4])
ax.set_yticklabels([pd.Timestamp(d).strftime('%b %d') for d in ds_subset.time.values[::4]])
fig.colorbar(im, label='OLR (W/m²)')
plt.show()