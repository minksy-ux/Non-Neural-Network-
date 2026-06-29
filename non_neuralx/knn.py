from collections import Counter
from typing import Optional

import numpy as np


class KNearestNeighbors:
    """k-NN classifier with Euclidean or Manhattan distance."""

    def __init__(self, k: int = 5, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def _distance(self, x: np.ndarray) -> np.ndarray:
        if self.X_train is None:
            raise RuntimeError("KNearestNeighbors is not fitted.")
        if self.metric == "euclidean":
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        if self.metric == "manhattan":
            return np.sum(np.abs(self.X_train - x), axis=1)
        raise ValueError(f"Unknown metric: {self.metric}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.y_train is None:
            raise RuntimeError("KNearestNeighbors is not fitted.")
        X = np.asarray(X)
        preds = []
        for x in X:
            dists = self._distance(x)
            idx = np.argsort(dists)[: self.k]
            preds.append(Counter(self.y_train[idx]).most_common(1)[0][0])
        return np.asarray(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
