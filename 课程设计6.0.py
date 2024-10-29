import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'C:\Users\Chen Yong\Downloads\archive\User.csv', skiprows = lambda x: x not in range(10), usecols = ['archive', 'fans', 'like_num'])
df = data.loc[:, :]  #取前1000行数据试试先
X = df['archive']
Y = df['fans']
Z = df['like_num']
result1 = X.corr(Y)
result2 = X.corr(Z)
print(result1, result2)  #两个相关系数
'''
sns.pairplot(df)
sns.pairplot(df , hue ='archive')  #两幅散点图


df.plot.scatter('like_num', 'fans', c='archive', colormap='jet')
plt.title('Correlation analysis')
'''
sns.set(font_scale=1.5,style="white")
lm = sns.lmplot(x = 'archive', y = 'fans', data = df, fit_reg = False)
lm.set(title = 'Correlation analysis')
lm.axes[0,0].set_xlim(-5,)
lm.axes[0,0].set_ylim(-5,)