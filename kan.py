import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [3, 2, 6, 5, 8, 2, 7, 8, 5, 3]
Z = [3, 4, 5, 2, 7, 5, 4, 4, 8, 2]
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.bar3d(X, Y, 0, 1, 1, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
