import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def accuracy_score(y_true: NDArray[np.int_], y_pred: NDArray[np.int_]) -> float:
    """Return the fraction of exact label matches."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def mean_squared_error(y_true: ArrayF, y_pred: ArrayF) -> float:
    """Return mean squared error between numeric vectors."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true_arr - y_pred_arr) ** 2))
