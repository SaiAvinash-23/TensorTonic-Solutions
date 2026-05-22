import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_std = np.std(x, axis=-1, keepdims=True)
    x_normalized = (x - x_mean) / (x_std + eps)
    return gamma * x_normalized + beta

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    batch_size, seq_len, d_model = Q.shape
    dk = d_model // num_heads

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, dk).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, dk).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, dk).transpose(0, 2, 1, 3)

    scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(dk)
    weights = softmax(scores, axis=-1)
    head_outputs = weights @ V_heads  # (batch, heads, seq, dk)

    concat = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    return concat @ W_o


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    ffn_out = np.matmul(np.maximum(0, np.matmul(x, W1) + b1), W2) + b2
    return ffn_out

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x_prime = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)
    output = layer_norm(x_prime + feed_forward(x_prime, W1, b1, W2, b2), gamma2, beta2)
    return output