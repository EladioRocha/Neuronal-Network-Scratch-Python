import numpy as np

class Activation_ReLU:
  def forward(self, inputs):
    ## Si hay números menores de cero, directamente los pasamos a 0.
    self.output = np.maximum(0, inputs)
    return self.output