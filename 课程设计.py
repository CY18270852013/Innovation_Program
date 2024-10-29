import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

data = pd.read_csv(r'C:\Users\Chen Yong\Downloads\archive\User.csv', skiprows = lambda x: x not in range(1000), usecols = ['archive', 'fans', 'like_num'])
df = data.loc[:, :]  #取前100000行数据试试先
X = df['archive']
Y = df['fans']
Z = df['like_num']
result1 = X.corr(Y)
result2 = X.corr(Z)
print(result1, result2)  #两个相关系数

sns.pairplot(df)
sns.pairplot(df , hue ='archive')  #两幅散点图