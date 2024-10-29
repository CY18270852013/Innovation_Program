import matplotlib.pyplot as plt
import numpy as np
scores = np.array([86, 77, 50, 98, 95, 65, 80, 40, 61, 82, 29, 13, 24, 0, 1, 2, 4, 5])
bins = [0, 20, 40, 60, 80, 100]
plt.hist(scores, rwidth = 0.95, color = 'y', edgecolor = 'r')