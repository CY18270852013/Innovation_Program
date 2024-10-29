# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:21:41 2024

@author: Chen Yong
"""

import tropycal.tracks as tracks
import tropycal.tornado as tornado
import datetime as dt
ibtracs_url=r'C:\Users\Chen Yong\Desktop\Research_and_cultivation_project\ibtracs.ALL.list.v04r00.csv'
 
ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True,ibtracs_url=r'C:\Users\Chen Yong\Desktop\Research_and_cultivation_project\ibtracs.ALL.list.v04r00.csv')
storm = ibtracs.get_storm(storm=('FREDDY',2023))
storm.plot()