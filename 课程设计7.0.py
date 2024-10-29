#导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

#实现绘图时汉字的显示
plt.rc('font', family='SimHei', size=13)

#以uid编号将2000000名用户划分为若干区间，然后在每个区间中从随机生成的行数开始抽取部分用户数据，最后共抽取10000组数据
df0 = pd.DataFrame()
a = 1
for i in range(20):
    b = random.randint(a,a+99499)
    df1 = pd.read_csv('User.csv',skiprows = b,nrows = 500)
    df1.columns = ['uid','name','avatar','level','sex','sign','vip_type','vip_status','vip_role','archive','fans','friend','like_num','is_senior']
    df0 = pd.concat([df0,df1],ignore_index = True)
    a+=100000

#提取出四个重要的列，并计算它们之间的相关系数
W = df0['uid']
X = df0['archive']
Y = df0['fans']
Z = df0['like_num']
result1 = X.corr(Y)
result2 = X.corr(Z)
result3 = W.corr(Y)
result4 = W.corr(Z)
print(result1, result2, result3, result4)  

#将整个数据集分成6份，并根据它们的不同列绘制散点图
df1 = df0.loc[:, ['uid', 'fans']]
df2 = df0.loc[:, ['uid', 'like_num']]
df3 = df0.loc[:, ['archive', 'fans']]
df4 = df0.loc[:, ['archive', 'like_num']]
df5 = df0.loc[:, ['fans', 'like_num']]
df6 = df0.loc[:, ['fans', 'like_num', 'archive']]

#绘制1和2号散点图，基于相同的x和y轴绘制多个散点图
sns.pairplot(df1)
sns.pairplot(df2)

#绘制3号散点图，使用线性回归模型绘制散点图，并设置边界
sns.set(font_scale = 1, style = "white")
lm3 = sns.lmplot(x = 'archive', y = 'fans', data = df3, fit_reg = False)
lm3.set(title = 'Correlation analysis between archive and fans')
lm3.axes[0,0].set_xlim(-5,)
lm3.axes[0,0].set_ylim(-5,)

#绘制4号散点图，与3号类似，使用线性回归模型绘制散点图，并设置边界
sns.set(font_scale = 1, style = "white")
lm4 = sns.lmplot(x = 'archive', y = 'like_num', data = df4, fit_reg = False)
lm4.set(title = 'Correlation analysis between archive and like_num')
lm4.axes[0,0].set_xlim(-5,)
lm4.axes[0,0].set_ylim(-5,)

#绘制5号散点图
sns.set(style = "ticks")  #通过调用 sns.set() 函数，对配色方案进行调整
pairplot = sns.pairplot(df5, diag_kind = "kde", plot_kws = dict(s = 4, edgecolor = "gray", linewidth = 0.5), diag_kws = dict(shade = True))  #通过设置 plot_kws 参数，对散点图的线宽和样式进行调整
sns.set(font_scale = 1.2)
fig5 = plt.gcf()
fig5.set_size_inches(10, 10)  #对画布大小进行调整
plt.subplots_adjust(top = 0.95)
#通过调用 seaborn 库自带的函数进行调整，实现了坐标轴标签、刻度、网格等属性的调整，提高了图像的可读性和解释性。
plt.suptitle("Correlation analysis between fans and like_num", fontsize = 18)
sns.despine()
pairplot.savefig("fig5.png", dpi=300)  #通过设置 dpi 参数，对图像每英寸点数进行调整

#绘制6号散点图
df6.plot.scatter('like_num', 'fans', c = 'archive', colormap='jet')
plt.title('Correlation analysis according to archive')

#自定义处理会员相关数据的函数
def vip(df):
    num1=len(df[df['vip_type']==0])
    num2=len(df[(df['vip_type']!=0) & (df['vip_status']==0)])
    num3=len(df[(df['vip_type']!=0) & (df['vip_status']==1)])
    data_vip = [['从未购入大会员',num1/10000],['曾经是大会员',num2/10000],['目前是大会员',num3/10000]]
    df_vip = pd.DataFrame(data_vip,columns=['状态','占比'])
    colors = ['cornflowerblue','orange','limegreen']
    plt.pie(df_vip['占比'],labels=df_vip['状态'],colors=colors,autopct='%.1f%%',textprops={'fontsize':'x-large'},labeldistance=1.2,startangle=90,shadow=True)
    plt.show()
    
    num4=len(df[df['vip_role']==0])
    num5=len(df[df['vip_role']==1])
    num6=len(df[df['vip_role']==3])
    num7=len((df[df['vip_role']==7]) | (df[df['vip_role']==15]))
    data_viprole = [['非大会员',num4/10000],['月度大会员',num5/10000],['年度大会员',num6/10000],['十年/百年大会员',num7/10000]]
    df_viprole = pd.DataFrame(data_viprole,columns=['会员类型','占比'])
    colors = ['lightcoral','lawngreen','deepskyblue','violet']
    plt.pie(df_viprole['占比'],labels=df_viprole['会员类型'],colors=colors,autopct='%.1f%%',textprops={'fontsize':'x-large'},labeldistance=1.2,startangle=90,shadow=True)
    plt.show()
    payment=25*(num5/10000)+233*(num6/10000)
    return(payment)   

#抽取10000名老用户的数据
df_oldusers = pd.DataFrame()
a = 1
for i in range(20):
    b = random.randint(a,a+49499)
    df1 = pd.read_csv('User.csv',skiprows=b,nrows=500)
    df1.columns = ['uid','name','avatar','level','sex','sign','vip_type','vip_status','vip_role','archive','fans','friend','like_num','is_senior']
    df_oldusers = pd.concat([df_oldusers,df1],ignore_index=True)
    a+=50000

#抽取10000名新用户的数据
df_newusers = pd.DataFrame()
a = 1000001
for i in range(20):
    b = random.randint(a,a+99499)
    df2 = pd.read_csv('User.csv',skiprows=b,nrows=500)
    df2.columns = ['uid','name','avatar','level','sex','sign','vip_type','vip_status','vip_role','archive','fans','friend','like_num','is_senior']
    df_newusers = pd.concat([df_newusers,df2],ignore_index=True)
    a+=100000 

#分别打印老用户和新用户在会员付费方面的贡献值
print(vip(df_oldusers))
print(vip(df_newusers))