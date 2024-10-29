import matplotlib.pyplot as plt
plt.figure (1)
plt.axes([.1, .2, 0.8, 0.8])
plt.plot(range(7), [3, 4, 7, 6, 2, 8, 9], color = 'r', marker = 'o')
plt.axes([.2, .4, 0.8, 0.5])
plt.plot(range(7), [5, 1, 8, 2, 6, 9, 4], color = 'green', marker = 'o')

