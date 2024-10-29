# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:44:24 2024

@author: Chen Yong
"""

import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'u_component_of_wind',
        'pressure_level': '100',
        'year': ['2023'],  # 或者你想分析的年份
        'month': [f'{i:02d}' for i in range(1, 13)],  # 1-12月
        'day': ['01'],  # 按月分析时，选每个月的第1天
        'time': '00:00',
        'area': [53.55, 73.66, 18.16, 135.05],  # [南, 西, 北, 东]，对应中国的范围
    },
    'ERA5_UV.nc'  # 输出文件名
)
