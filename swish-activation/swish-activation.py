import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.array(x, dtype=np.float64)
    sigmoid = 1 / (1 + np.exp(-x))
    swiss = x * sigmoid
    return swiss