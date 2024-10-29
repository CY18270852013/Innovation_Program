# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:24:54 2024

@author: Chen Yong
"""

import xarray as xr

# 读取IBTrACS数据集
ibtracs_data = xr.open_dataset(r'C:\Users\Chen Yong\Downloads\IBTrACS.ALL.v04r00.nc')

# 获取所有变量名称
variables = ibtracs_data.variables

# 打印所有变量名称
for var in variables:
    print(var)
