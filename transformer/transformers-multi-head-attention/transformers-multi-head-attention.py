import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:

    batch, seq_len, d_model = Q.shape[0], Q.shape[1], Q.shape[2]  # 1, 3, 4
    dk = d_model // num_heads                                       # 4//2 = 2

    # Step 1 — project Q, K, V
    Q_proj = Q @ W_q        # (1,3,4) @ (4,4) = (1,3,4)
    K_proj = K @ W_k        # (1,3,4) @ (4,4) = (1,3,4)
    V_proj = V @ W_v        # (1,3,4) @ (4,4) = (1,3,4)

    # Step 2 — split into heads (keep batch dim!) 
    Q_heads = Q_proj.reshape(batch, seq_len, num_heads, dk).transpose(0, 2, 1, 3)  # (1,2,3,2)
    K_heads = K_proj.reshape(batch, seq_len, num_heads, dk).transpose(0, 2, 1, 3)  # (1,2,3,2)
    V_heads = V_proj.reshape(batch, seq_len, num_heads, dk).transpose(0, 2, 1, 3)  # (1,2,3,2)

    # Step 3 — scaled dot product attention
    scores      = Q_heads @ K_heads.transpose(0, 1, 3, 2)  # (1,2,3,3)
    scores      = scores / np.sqrt(dk)                      # scale
    weights     = softmax(scores, axis=-1)                  # (1,2,3,3)
    head_outputs = weights @ V_heads                        # (1,2,3,2)

    # Step 4 — concatenate heads
    concat = head_outputs.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)  # (1,3,4)

    # Step 5 — output projection
    output = concat @ W_o   # (1,3,4)

    return output
