import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    similarity_scores = torch.matmul(Q, K.transpose(-2, -1))
    scaled_down_scores = similarity_scores / math.sqrt(K.shape[-1])
    probs = F.softmax(scaled_down_scores, dim= -1)
    weighted_sum = torch.matmul(probs, V)

    return weighted_sum