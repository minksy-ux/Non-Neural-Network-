from typing import Tuple

import numpy as np


def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 6,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic classification data."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))

    if n_classes == 2:
        decision = X[:, 0] + 0.7 * X[:, 1] - 0.45 * X[:, 2] + 0.3 * rng.normal(size=n_samples)
        y = (decision > 0.0).astype(int)
    else:
        raw = X[:, : min(3, n_features)] @ np.arange(1, min(3, n_features) + 1)
        raw += 0.3 * rng.normal(size=n_samples)
        bins = np.quantile(raw, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(raw, bins)

    return X, y


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))
