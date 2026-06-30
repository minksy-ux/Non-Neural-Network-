import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def pairwise_squared_distance(X: ArrayF) -> ArrayF:
    """Return dense pairwise squared Euclidean distance matrix."""
    X_arr = np.asarray(X, dtype=float)
    return np.sum((X_arr[:, None] - X_arr[None, :]) ** 2, axis=-1)
