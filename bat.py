import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

N = 5
ind = np.arange(N)
stu1 = np.array([15, 13, 8, 10, 4])
stu2 = np.array([18, 12, 11, 4, 6])
plt.bar(ind, stu1, width = 0.3, label = '班级1')
plt.bar(ind + 0.3, stu2, width = 0.3, label = '班级2')
plt.title('两班成绩对比')
plt.xlabel('等级')
plt.ylabel('数量')
plt.xticks(ind, ('A', 'B', 'C', 'D', 'E'))
plt.yticks(np.arange(0, 41, 5))

plt.legend(loc = 'best')
