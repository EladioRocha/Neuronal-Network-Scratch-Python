import numpy as np

class Loss:
  ## Recibe valores de muestra y el valor real que le corresponde.
  def calculate(self, output, y):
    ## Calcular el error de las muestras.
    sample_losses = self.forward(output, y)
    data_loss = np.mean(sample_losses)
    return data_loss

class Loss_CategoricalCrossentropy(Loss):
  def forward(self, y_pred, y_true):
    ## Número de muestras
    samples = len(y_pred)
    
    ## Evita una división entre cero debido a que de otra forma podría darnos cero log(0)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    ##Probabilidad de los target values
    if len(y_true.shape) == 1:
      correct_confidences = y_pred_clipped[range(samples), y_true]
    elif len(y_true.shape) == 1:
      correct_confidences = np.sum(
        y_pred_clipped * y_true,
        axis = 1
      )

    # Guarda y regresa las perdidas.
    negative_log_likelihoods = -np.log(correct_confidences)
    return negative_log_likelihoods