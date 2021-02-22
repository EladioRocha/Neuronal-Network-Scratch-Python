import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from ActivationReLU import Activation_ReLU
from ActivationSoftmax import Activation_Softmax

nnfs.init()

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		print('biases', self.biases)
	
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases
		return self.output

x, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
dense_output = dense1.forward(x)

activation1 = Activation_ReLU()
activation_output = activation1.forward(dense_output)

softmax = Activation_Softmax()
softmax.forward([[1, 2, 3]])

print('============== TEST WITH SPERIAL DATA =============')
x, y = spiral_data(samples = 100, classes = 3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])