from datetime import time
from os import times
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
tcFile =r'C:\Users\Chen Yong\Desktop\Research_and_cultivation_project\IBTrACS.ALL.v04r00.nc'
infile = xr.open_dataset(tcFile)

# read variables
lat  = infile['usa_lat']          # storm center latitude,从infile中提取热带气旋中心的纬度数据，并将其存储在变量lat中。
lon  = infile['usa_lon']         # storm center longitude,从infile中提取热带气旋中心的经度数据，并将其存储在变量lon中。          # 2 = WP - western north Pacific
stormYear = infile['season'].data              # year based on season,从infile中提取热带气旋的年份数据，并将其存储在变量stormYear中。
sid=infile['sid']   #从infile中提取热带气旋的ID数据，并将其存储在变量sid中。
number=infile['number'].data    #从infile中提取热带气旋的编号数据，并将其存储在变量number中。
time2d = infile['iso_time'].data    #从infile中提取热带气旋的时间数据，并将其存储在变量time2d中。     # time step
name=infile['name'].data     # 从infile中提取热带气旋的名称数据，并将其存储在变量name中。
basin=infile['basin'].data     #从infile中提取热带气旋的海盆数据，并将其存储在变量basin中
basin1=basin[:,0]    #从basin中取出第一列数据，并将其存储在变量basin1中。
tracktype=infile['track_type'].data     #从infile中提取热带气旋的路径类型数据，并将其存储在变量tracktype中。
#date_time = infile['date_time'].data

tcmask = np.where(sid==b'2023037S12119')  #使用np.where函数根据条件过滤符合条件的热带气旋数据，并将结果存储在变量tcmask中。

latselect  = lat[tcmask]    #根据tcmask中的索引，从lat变量中选取符合条件的纬度数据，并将结果存储在变量latselect中。
lonselect  = lon[tcmask]    #根据tcmask中的索引，从lon变量中选取符合条件的经度数据，并将结果存储在变量lonselect中。
nameselect = name[tcmask]   #根据tcmask中的索引，从name变量中选取符合条件的名称数据，并将结果存储在变量nameselect中。
timeselect = time2d[tcmask]     #根据tcmask中的索引，从time2d变量中选取符合条件的时间数据，并将结果存储在变量timeselect中。
numberselect = number[tcmask]   #根据tcmask中的索引，从number变量中选取符合条件的编号数据，并将结果存储在变量numberselect中。
basinselect= basin[tcmask]  #根据tcmask中的索引，从basin变量中选取符合条件的海盆数据，并将结果存储在变量basinselect中。
#dateselect = date_time[tcmask]

#print(len(numberselect))    #输出符合条件的热带气旋数量。

#此处我们可以输出一个数据，我们就可以准确知道满足年份地区和轨道类型的台风的具体个数，而不需要自己数
latselect1=np.array(latselect)  #将latselect转换为numpy数组，存储在变量latselect1中。
lonselect1=np.array(lonselect)  #将lonselect转换为numpy数组，存储在变量lonselect1中。
#dateselect1=np.array(dateselect)

fig=plt.figure(figsize=(20,10)) #设置画布大小
 
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))   #设置投影方式,在图形对象上创建一个地图投影为PlateCarree的坐标系
# Set figure extent & ticks
ax.coastlines() #在地图上添加海岸线。
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)    #在地图上添加网格线，并设置参数以绘制经纬度标签。
ax.set_extent([30,120, -25, -5])  #设置地图的范围，纬度范围为-30到-10，经度范围为30到120。
ax.add_feature(cfeature.OCEAN)  #在地图上添加海洋特征
ax.set_title('The first stage path of tropical cyclone Freddy',fontsize=20, loc='center')   #设置地图标题，显示符合条件的热带气旋数量。
ax.add_feature(cfeature.LAND, edgecolor='b')    #在地图上添加陆地特征，并设置边缘颜色为蓝色。
for i in range(0,len(numberselect)):    #遍历符合条件的热带气旋数量
    tclat=latselect1[i,:]   #获取第i个热带气旋的纬度数据。
    tclon=lonselect1[i,:]   # 获取第i个热带气旋的经度数据。
    #dates = date_time[i,]
    ax.plot(tclon,tclat,color='red',linewidth=1,transform=ccrs.PlateCarree())   #在地图上绘制热带气旋路径，使用黑色线条，设置线宽为0.5。
    cb = ax.scatter(tclon,tclat,s=5,transform=ccrs.PlateCarree())   #继续未完成的代码，似乎是在绘制散点图，但这里缺少后续的参数设置。
    #for j in range(len(tclon)):
        #ax.text(tclon[j], tclat[j], str(dates), color='red', fontsize=8, transform=ccrs.PlateCarree())