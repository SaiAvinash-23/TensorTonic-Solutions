import numpy as np

def sample_var_std(x):
    """
    Returns: dict with 'variance' and 'std_dev' as floats.
    """
    x = np.array(x, dtype=np.float64)
    var = np.var(x, ddof=1)
    std = var**0.5
    return {
        "variance": var,
        "std_dev": std
    }