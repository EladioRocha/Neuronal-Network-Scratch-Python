import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
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

############################################## INICIO DE SPIRAL DATA #####################################################
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
print('Loss spiral: ', loss)

predictions = np.argmax(activation2.output, axis = 1)
if(len(y.shape) == 2):
	y = np.argmax(y, axis = 1)

accuracy = np.mean(predictions == y)

print('Accuracy spiral: ', accuracy)
############################################## FINAL DE SPIRAL DATA #####################################################

## Usando clase Loss en ejemplo individual.
softmax_outputs = np.array([[0.7, 0.1, 0.2],
									 [0.1, 0.5, 0.4],
									 [0.02, 0.9, 0.08]
									])

class_targets = np.array([0, 1, 1])

loss = loss_function.calculate(softmax_outputs, class_targets)
print('Loss: ', loss)

## Accuracy calculation

## Ejemplo de 3 muestras
softmax_outputs = np.array([
	[0.7, 0.2, 0.1],
	[0.5, 0.1, 0.4],
	[0.02, 0.9, 0.08]
])

# Etiqueta de las 3 muestras
class_targets = np.array([0, 1, 1])
# Obtiene los valores máximos de cada posicion -> [0 0 1]
predictions = np.argmax(softmax_outputs, axis = 1)
if len(class_targets.shape) == 2:
  class_targets = np.argmax(class_targets, axis=1)

accuracy = np.mean(predictions == class_targets)
print('Accuracy: ', accuracy)

########################## EJEMPLOS USANDO DATA VERTICAL ######################################
X, y = vertical_data(samples = 100, classes = 3)
dense1 = Layer_Dense(2, 3) ## primera capa 2 entradas
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3) ## 3 entradas, 3 salidas
activation2 = Activation_Softmax()

## Función de perdida
loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 9999999  # Valor inicial
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
	# Actualizar los pesos con algunos valores pequeños
	dense1.weights += 0.05 * np.random.randn(2, 3)
	dense1.biases += 0.05 * np.random.randn(1, 3)
	dense2.weights += 0.05 * np.random.randn(3, 3)
	dense2.biases += 0.05 * np.random.randn(1, 3)

	## Pasa a las siguientes capas
	dense1.forward(X)
	activation1.forward(dense1.output)
	dense2.forward(activation1.output)
	activation2.forward(dense2.output)

	# calculo del error con la salidad de la segunda capa de activacion y pasandole las labels
	loss = loss_function.calculate(activation2.output, y)


	# Calculo de la precisión con los datos de la capa de salida
	predictions = np.argmax(activation2.output, axis=1)
	accuracy = np.mean(predictions==y)

	# Si la perdida es menor, guarda los valores y se imprimen en pantalla
	if loss < lowest_loss:
			print('New set of weights found, iteration:', iteration,
						'loss:', loss, 'acc:', accuracy)
			best_dense1_weights = dense1.weights.copy()
			best_dense1_biases = dense1.biases.copy()
			best_dense2_weights = dense2.weights.copy()
			best_dense2_biases = dense2.biases.copy()
			lowest_loss = loss
	# Revierte los pesos y bias pues los valores no fueron mejores
	else:
			dense1.weights = best_dense1_weights.copy()
			dense1.biases = best_dense1_biases.copy()
			dense2.weights = best_dense2_weights.copy()
			dense2.biases = best_dense2_biases.copy()

## 149