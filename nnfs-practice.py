import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data


x, y = spiral_data(samples = 100, classes = 3)
print(len(y))
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = 'brg')
plt.show()

X, y = vertical_data(samples = 100, classes = 3)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 40, cmap = 'brg')
plt.show()