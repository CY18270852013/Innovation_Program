import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

df0 = pd.DataFrame()
a = 1
for i in range(20):
    b = random.randint(a,a+99499)
    df1 = pd.read_csv('User.csv',skiprows=b,nrows=500)
    df1.columns = ['uid','name','avatar','level','sex','sign','vip_type','vip_status','vip_role','archive','fans','friend','like_num','is_senior']
    df0 = pd.concat([df0,df1],ignore_index=True)
    a+=100000

W = df0['uid']
X = df0['archive']
Y = df0['fans']
Z = df0['like_num']

#四个相关系数
result1 = X.corr(Y)
result2 = X.corr(Z)
result3 = W.corr(Y)
result4 = W.corr(Z)
print(result1, result2, result3, result4)  

#6幅散点图
df1 = df0.loc[:, ['uid', 'fans']]
df2 = df0.loc[:, ['uid', 'like_num']]
df3 = df0.loc[:, ['archive', 'fans']]
df4 = df0.loc[:, ['archive', 'like_num']]
df5 = df0.loc[:, ['fans', 'like_num']]
df6 = df0.loc[:, ['fans', 'like_num', 'archive']]
sns.pairplot(df1)
sns.pairplot(df2)
sns.pairplot(df3)
sns.pairplot(df4)
sns.pairplot(df5)
sns.pairplot(df6, hue ='archive')