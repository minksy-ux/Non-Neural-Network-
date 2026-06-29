from collections import Counter
from typing import Optional, Tuple

import numpy as np


class DecisionTree:
    """Entropy-based decision tree classifier."""

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: Optional[dict] = None

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def _information_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        n = len(parent)
        if n == 0:
            return 0.0
        w_left = len(left) / n
        w_right = len(right) / n
        return self._entropy(parent) - (w_left * self._entropy(left) + w_right * self._entropy(right))

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)

        return best_feature, best_threshold, float(best_gain)

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            return {"leaf": True, "class": Counter(y).most_common(1)[0][0]}

        feature, threshold, gain = self._find_best_split(X, y)
        if feature is None or gain <= 0:
            return {"leaf": True, "class": Counter(y).most_common(1)[0][0]}

        left_mask = X[:, feature] <= threshold
        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.tree = self._build(np.asarray(X), np.asarray(y))
        return self

    def _predict_one(self, x: np.ndarray, node: dict):
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("DecisionTree is not fitted.")
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.tree) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
