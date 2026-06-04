import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Returns: dict with 'mean', 'median', 'mode' as floats.
    """
    counts = Counter(x)
    mode = counts.most_common(1)[0][0]
    return {"mean": np.mean(x), 
           "median": np.median(x),
           "mode": mode}