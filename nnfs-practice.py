import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

x, y = spiral_data(samples = 100, classes = 3)
print(len(y))
plt.scatter(x[:, 0], x[:, 1], c = y, cmap = 'brg')
plt.show()