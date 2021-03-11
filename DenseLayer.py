import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from ActivationReLU import Activation_ReLU
from ActivationSoftmax import Activation_Softmax
from Loss import Loss_CategoricalCrossentropy

nnfs.init()

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		## Pesos iniciales
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		## Bias iniciales, en este caso serán iniciados en 0
		self.biases = np.zeros((1, n_neurons))
	
	def forward(self, inputs):
		print('PESOS:', self.weights.shape, inputs.shape)
		print(inputs[3])
		self.output = np.dot(inputs, self.weights) + self.biases
		print(self.output[3])
		return self.output

x, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
dense_output = dense1.forward(x)
activation1 = Activation_ReLU()
activation_output = activation1.forward(dense_output)

softmax = Activation_Softmax()
softmax.forward([[1, 2, 3]])

print('============== EJEMPLO USANDO LAS FUNCIONES CON MUESTRAS DE SPIRAL DATA. =============')
x, y = spiral_data(samples = 100, classes = 3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
## Softmax resultados
print(activation2.output[:5])
# Calculando la perdida en el ejemplo con la data de spiral.
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print('Loss: ', loss)

## Usando clase Loss en ejemplo individual.
softmax_outputs = np.array([[0.7, 0.1, 0.2],
									 [0.1, 0.5, 0.4],
									 [0.02, 0.9, 0.08]
									])

class_targets = np.array([0, 1, 1])

loss = loss_function.calculate(softmax_outputs, class_targets)
print(loss)