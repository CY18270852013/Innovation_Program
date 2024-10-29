import matplotlib.pyplot as plt

x = ['Mon.', 'Tues.', 'Wed.', 'Thur.', 'Fri.', 'Sat.', 'Sun.']
y1 = [10, 10, 12, 10, 8, 6, 8]
y2 = [1, 1, 1, 1.5, 2, 3, 2]
y3 = [2, 2, 3, 2, 3, 4, 3]
labels = ['study time', 'sports time', 'fun time']
plt.stackplot(x, y1, y2, y3, labels = labels)
plt,title('Daily Schedule Stacked Area Chart')
plt.legend(loc = 'best')