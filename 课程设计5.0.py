import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv(r'C:/Users/Chen Yong/Downloads/archive/User.csv', skiprows=lambda x: x not in range(10), usecols=['archive', 'fans', 'like_num'])
df = data.loc[:, :] # 取前100000行数据试试先
X = df['archive']
Y = df['fans']
Z = df['like_num']
result1 = X.corr(Y)
result2 = X.corr(Z)
print(result1, result2) # 两个相关系数

sns.set(style="ticks")  #通过调用 sns.set() 函数，对配色方案进行调整
pairplot = sns.pairplot(df, diag_kind="kde", plot_kws=dict(s=4, edgecolor="gray", linewidth=0.5), diag_kws=dict(shade=True), corner=True)  #通过设置 plot_kws 参数，对散点图的线宽和样式进行调整
#通过设置 diag_kind 和 corner 参数，对子图类型和布局进行调整
sns.set(font_scale=0.8)
fig = plt.gcf()
fig.set_size_inches(10, 10)  #对画布大小进行调整
plt.subplots_adjust(top=0.95)
#通过调用 seaborn 库自带的函数进行调整，实现了坐标轴标签、刻度、网格等属性的调整，提高了图像的可读性和解释性。
plt.suptitle("Pairplot of User Dataset", fontsize = 18)
sns.despine()
pairplot.savefig("User_Pairplot.png", dpi=300)  #通过设置 dpi 参数，对图像每英寸点数进行调整