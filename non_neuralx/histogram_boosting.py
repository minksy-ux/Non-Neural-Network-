from typing import List, Optional, Tuple

import numpy as np


class HistogramGradientBoosting:
    """Compact histogram-style gradient boosting for binary classification."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: List[dict] = []
        self.base_score: float = 0.0

    def _best_split(self, X: np.ndarray, gradients: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left = X[:, feature] <= threshold
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue

                g_left = gradients[left].sum()
                g_right = gradients[right].sum()
                gain = (g_left ** 2) / (left.sum() + 1e-12) + (g_right ** 2) / (right.sum() + 1e-12)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)

        return best_feature, best_threshold, float(best_gain)

    def _build_tree(self, X: np.ndarray, gradients: np.ndarray, depth: int = 0):
        if depth >= self.max_depth or len(X) < 8:
            return float(np.mean(gradients))

        feature, threshold, gain = self._best_split(X, gradients)
        if feature is None or gain <= 0:
            return float(np.mean(gradients))

        left = X[:, feature] <= threshold
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left], gradients[left], depth + 1),
            "right": self._build_tree(X[~left], gradients[~left], depth + 1),
        }

    def _predict_tree_one(self, x: np.ndarray, tree) -> float:
        if not isinstance(tree, dict):
            return float(tree)
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree_one(x, tree["left"])
        return self._predict_tree_one(x, tree["right"])

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistogramGradientBoosting":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._classes = np.unique(y)
        if len(self._classes) != 2:
            raise ValueError("HistogramGradientBoosting supports binary classification only.")

        y_bin = np.where(y == self._classes[0], 0.0, 1.0)
        p0 = np.clip(y_bin.mean(), 1e-6, 1 - 1e-6)
        self.base_score = float(np.log(p0 / (1 - p0)))
        raw = np.full(len(y_bin), self.base_score)

        self.trees = []
        for _ in range(self.n_estimators):
            prob = self._sigmoid(raw)
            gradients = y_bin - prob
            tree = self._build_tree(X, gradients)
            self.trees.append(tree)
            update = np.array([self._predict_tree_one(x, tree) for x in X])
            raw += self.learning_rate * update

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        raw = np.full(X.shape[0], self.base_score)
        for tree in self.trees:
            raw += self.learning_rate * np.array([self._predict_tree_one(x, tree) for x in X])
        p1 = self._sigmoid(raw)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.where(probs[:, 1] >= 0.5, self._classes[1], self._classes[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
