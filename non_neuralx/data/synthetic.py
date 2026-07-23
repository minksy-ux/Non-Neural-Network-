from typing import Tuple

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int_]


def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 6,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[ArrayF, ArrayI]:
    """Create deterministic synthetic classification data."""
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
