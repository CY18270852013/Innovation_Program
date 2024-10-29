# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:47:22 2024

@author: Chen Yong
"""

import os
import maskout
from netCDF4 import Dataset
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

##########################################################
## 读数据
f = Dataset(r'C:\Users\Chen Yong\Downloads\PV_2007_2016.nc')
lat = f.variables['latitude'][:] 
lon = f.variables['longitude'][:]
t = f.variables['t'][2,1,:,:]  # 气温
##########################################################
#白化中国地图，加南海九段线，加海岸线
SHP = r'C:\Users\qiuyu\.local\share\cartopy\shapefiles\natural_earth\cultural\china_shp'


def make_map(ax,box,lon,lat,var,proj,title,if_coast, if_nanhai):
    projection = ccrs.PlateCarree()
    # 加国界
    ax.add_geometries(Reader(os.path.join(SHP, 'cnmap.shp')).geometries(),
                      ccrs.PlateCarree(),facecolor='none',edgecolor='k', linewidth=0.7)
    # 加海岸线
    if if_coast:
        ax.add_geometries(Reader(os.path.join(SHP, 'coastline.shp')).geometries(),
                        ccrs.PlateCarree(),facecolor='none',edgecolor='k', linewidth=0.7)
    #标注坐标轴
    ax.set_extent([box[0],box[1],box[2],box[3]])
    ax.set_xticks(np.linspace(box[0], box[1],5), crs=projection) 
    ax.set_yticks(np.linspace(box[2], box[3],5), crs=projection)
    #zero_direction_label=True 有度的标识，False则去掉'''
    lon_formatter = LongitudeFormatter(zero_direction_label=True) 
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    #添加网格线
    ax.gridlines(linestyle='--',alpha=0.4)   

    # plot 
    cf = ax.contourf(lon,lat,var,cmap = mpl.cm.RdBu_r,
                     transform=ccrs.PlateCarree())  
    plt.colorbar(cf,ax=ax, extend='both',orientation='vertical') 
    maskout.shp2clip(cf,ax,shpfile=os.path.join(SHP, 'country1.shp'),region='China',proj= proj)
    ax.set_title(title)
    return ax

# make plot 

box1=[70,140,0,50]
box2=[70,140,15,50]
proj=ccrs.PlateCarree()
fig = plt.figure(figsize=(15,7))

ax1 = fig.add_subplot(221,projection = ccrs.PlateCarree())
ax2 = fig.add_subplot(222,projection = ccrs.PlateCarree())
ax3 = fig.add_subplot(223,projection = ccrs.PlateCarree())
ax4 = fig.add_subplot(224,projection = ccrs.PlateCarree())

make_map(ax1,box1,lon,lat,t,proj,title='(a) With coastline',if_coast=True, if_nanhai=True)
make_map(ax2,box2,lon,lat,t,proj,title='(b) With coastline + Nanhai',if_coast=True, if_nanhai=False)
make_map(ax3,box1,lon,lat,t,proj,title='(c) Without coastline',if_coast=False, if_nanhai=False)
make_map(ax4,box2,lon,lat,t,proj,title='(d) Without coastline + Nanhai',if_coast=False, if_nanhai=True)

#----------添加南海小地图------------------
def add_nanhai (ax,pos,if_coast):
    #--------------右下角添加南海地图------------------------------------------
    box_nanhai=[103,125,2,25]
    ax_nanhai = fig.add_axes(pos,projection = ccrs.PlateCarree())
    # 加国界
    ax_nanhai.add_geometries(Reader(os.path.join(SHP, 'cnmap.shp')).geometries(),
                      ccrs.PlateCarree(),facecolor='none',edgecolor='k', linewidth=0.7)
    if if_coast:
        ax_nanhai.add_geometries(Reader(os.path.join(SHP, 'coastline.shp')).geometries(),
                        ccrs.PlateCarree(),facecolor='none',edgecolor='k', linewidth=0.7)
    ax_nanhai.set_extent([box_nanhai[0],box_nanhai[1],box_nanhai[2],box_nanhai[3]])

pos1 = [0.757, 0.54, 0.1, 0.1]
pos2 = [0.757, 0.124, 0.1, 0.1]
add_nanhai(ax2,pos1,if_coast=True)
add_nanhai(ax4,pos2,if_coast=False)

plt.savefig('map.png')
