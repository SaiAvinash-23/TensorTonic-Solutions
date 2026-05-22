import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    ffn_x = np.matmul(np.maximum(0, (np.matmul(x, W1) + b1)), W2) + b2
    return ffn_x