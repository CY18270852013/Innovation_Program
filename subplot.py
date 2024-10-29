import matplotlib.pyplot as plt
plt.figure (1)
plt.subplot(211)
plt.plot(range(7), [3, 4, 7, 6, 2, 8, 9], color = 'r', marker = 'o')
plt.subplot(212)
plt.plot(range(7), [5, 1, 8, 2, 6, 9, 4], color = 'green', marker = 'o')

