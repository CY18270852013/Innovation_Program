import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = [item[0] for item in iris.data]
Y = [item[2] for item in iris.data]
plt.scatter(X[:50], Y[:50], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100], Y[50:100], color = 'green', marker = '*', label = 'versicolor')

plt.scatter(X[-50:], Y[-50:], color = 'blue', marker = 'D', label = 'virginica')
plt.title('Iris Dataset Scatter Chart')
plt.legend(loc = 'best')
