# Neural Networks from Scratch — Python Exercises

Study exercises following *Neural Networks from Scratch in Python* by **Harrison Kinsley and Daniel Kukieła**. The implementations are derived from the book's practice material; this repository is a learning companion, not an independently authored neural-network framework.

## Setup

Use Python 3 and install the dependencies in your development environment:

```sh
python -m pip install numpy matplotlib nnfs
```

No dependency versions are pinned. Plotting requires an environment with a working Matplotlib display backend.

## Suggested reading order

| File | Topics |
| --- | --- |
| [nnfs-practice.py](nnfs-practice.py) | Generate and plot spiral and vertical datasets. |
| [ActivationReLU.py](ActivationReLU.py) | ReLU forward pass. |
| [ActivationSoftmax.py](ActivationSoftmax.py) | Numerically stabilized softmax forward pass. |
| [Loss.py](Loss.py) | Mean loss and categorical cross-entropy. |
| [DenseLayer.py](DenseLayer.py) | Dense layers, forward passes, accuracy, and random weight-search examples. |

Start with the dataset visualization:

```sh
python nnfs-practice.py
```

Close the first plot to continue to the next. To run the network exercises, use `python DenseLayer.py`. That file runs examples at module scope, including a 10,000-iteration random weight search and extensive debug output; importing it also executes those examples.

## Scope and known limitations

The code explores forward propagation and random perturbations of weights, rather than a complete backpropagation training pipeline. In `Loss.py`, the one-hot-target branch repeats the sparse-target condition, so it does not handle two-dimensional target arrays as intended. Use the sparse-label examples when studying the current implementation.

There is no automated test suite or reproducible training benchmark. Documentation checks do not establish model accuracy. The book attribution above is preserved from the original README; the project is not presented as the official NNFS package.
