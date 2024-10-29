import numpy as np
import matplotlib.pyplot as plt
t = np.arange(0, 4, 0.1)
plt.figure(figsize = (8, 6), dpi = 500)
plt.plot(t, t, color = 'blue', linestyle = '-', linewidth = 3, label = 'line1')
plt.plot(t, t + 2, color = 'magenta', linestyle = '', marker = '*', linewidth = 3, label = 'line2')
plt.plot(t, t**2, color = 'pink', linestyle = '', marker = '+', linewidth = 3, label = 'line3')
plt.legend(loc = 'best')
plt.title('my first')
plt.xlabel('haha')
plt.ylabel('yoxi')