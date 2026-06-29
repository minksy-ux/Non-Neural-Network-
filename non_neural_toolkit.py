"""
NON-NEURAL MACHINE LEARNING TOOLKIT

Implements classical machine learning models without neural networks:
- Decision Tree (classification)
- k-Nearest Neighbors (classification)
- Linear SVM (binary classification)
- Linear Regression (regression)
- Spectral Graph Pruning Classifier (novel graph-based classifier)
- Histogram-style Gradient Boosting (classification)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np


class DecisionTree:
    """Simple classification tree using information gain (entropy)."""

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
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
        ):
            counts = Counter(y)
            return {"leaf": True, "class": counts.most_common(1)[0][0]}

        feature, threshold, gain = self._find_best_split(X, y)
        if feature is None or gain <= 0:
            counts = Counter(y)
            return {"leaf": True, "class": counts.most_common(1)[0][0]}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.tree = self._build(np.asarray(X), np.asarray(y))
        return self

    def _predict_one(self, x: np.ndarray, node: dict) -> int:
        if node["leaf"]:
            return int(node["class"])
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
        preds: List[int] = []
        for x in X:
            dists = self._distance(x)
            nn_idx = np.argsort(dists)[: self.k]
            preds.append(Counter(self.y_train[nn_idx]).most_common(1)[0][0])
        return np.asarray(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class LinearSVM:
    """Linear SVM (binary) trained with simple sub-gradient descent."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, lambda_param: float = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("LinearSVM currently supports binary classification only.")

        y_bin = np.where(y == classes[0], -1, 1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                xi = X[i]
                margin = y_bin[i] * (np.dot(xi, self.weights) - self.bias)
                if margin >= 1:
                    dw = 2 * self.lambda_param * self.weights
                    db = 0.0
                else:
                    dw = 2 * self.lambda_param * self.weights - y_bin[i] * xi
                    db = -y_bin[i]
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        self._classes = classes
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("LinearSVM is not fitted.")
        raw = np.dot(np.asarray(X), self.weights) - self.bias
        pred_bin = np.where(raw >= 0, 1, -1)
        return np.where(pred_bin == -1, self._classes[0], self._classes[1])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class LinearRegression:
    """Ordinary least squares linear regression."""

    def __init__(self):
        self.intercept: Optional[float] = None
        self.coefficients: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_i = np.column_stack([np.ones(X.shape[0]), X])
        # Pseudo-inverse is numerically stable and handles singular matrices.
        beta = np.linalg.pinv(X_i) @ y
        self.intercept = float(beta[0])
        self.coefficients = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients is None or self.intercept is None:
            raise RuntimeError("LinearRegression is not fitted.")
        return self.intercept + np.dot(np.asarray(X), self.coefficients)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))


class SpectralGraphPruningClassifier:
    """
    Novel non-neural classifier:
    1) Build weighted k-NN graph
    2) Compute spectral embedding from normalized Laplacian
    3) Prune weak spectral-coherence edges
    4) Run label propagation
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        n_components: int = 5,
        prune_threshold: float = 0.25,
        n_iterations: int = 20,
        random_state: int = 42,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.prune_threshold = prune_threshold
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.embedding: Optional[np.ndarray] = None
        self.pruned_graph: Optional[np.ndarray] = None

    def _pairwise_sq_dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
        a2 = np.sum(A * A, axis=1, keepdims=True)
        b2 = np.sum(B * B, axis=1, keepdims=True).T
        d2 = a2 + b2 - 2.0 * (A @ B.T)
        return np.maximum(d2, 0.0)

    def _knn_graph(self, X: np.ndarray, n_neighbors: Optional[int] = None) -> np.ndarray:
        n_neighbors = n_neighbors or self.n_neighbors
        n = X.shape[0]
        d2 = self._pairwise_sq_dist(X, X)
        np.fill_diagonal(d2, np.inf)

        idx = np.argpartition(d2, kth=min(n_neighbors, n - 1), axis=1)[:, :n_neighbors]
        A = np.zeros((n, n), dtype=float)
        rows = np.arange(n)
        for i in rows:
            nbrs = idx[i]
            weights = np.exp(-d2[i, nbrs])
            A[i, nbrs] = weights
        A = np.maximum(A, A.T)
        return A

    @staticmethod
    def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
        deg = np.sum(A, axis=1)
        inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
        D_inv_sqrt = np.diag(inv_sqrt)
        I = np.eye(A.shape[0])
        return I - D_inv_sqrt @ A @ D_inv_sqrt

    def _spectral_embedding(self, A: np.ndarray) -> np.ndarray:
        L = self._normalized_laplacian(A)
        vals, vecs = np.linalg.eigh(L)
        # Skip the first eigenvector (trivial component)
        comps = min(self.n_components, max(1, A.shape[0] - 1))
        return vecs[:, 1 : 1 + comps]

    def _prune(self, A: np.ndarray, emb: np.ndarray) -> np.ndarray:
        row, col = np.where(A > 0)
        coh = np.sum((emb[row] - emb[col]) ** 2, axis=1)
        thr = self.prune_threshold * (np.mean(coh) + 0.5 * np.std(coh))
        keep = coh < thr

        P = np.zeros_like(A)
        P[row[keep], col[keep]] = A[row[keep], col[keep]]
        P = np.maximum(P, P.T)
        return P

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralGraphPruningClassifier":
        np.random.seed(self.random_state)
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)

        graph = self._knn_graph(self.X_train)
        self.embedding = self._spectral_embedding(graph)
        self.pruned_graph = self._prune(graph, self.embedding)
        return self

    def _label_propagation(self, A: np.ndarray, labels: np.ndarray, n_train: int) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")

        Y = np.zeros((A.shape[0], len(self.classes_)), dtype=float)
        for i, c in enumerate(self.classes_):
            Y[labels == c, i] = 1.0

        train_indices = np.arange(n_train)
        clamp = np.eye(len(self.classes_))[np.searchsorted(self.classes_, labels[:n_train])]

        row_sum = A.sum(axis=1, keepdims=True)
        A_norm = A / np.maximum(row_sum, 1e-12)

        for _ in range(self.n_iterations):
            Y = A_norm @ Y
            Y[train_indices] = clamp

        return Y

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None or self.pruned_graph is None:
            raise RuntimeError("SpectralGraphPruningClassifier is not fitted.")

        X = np.asarray(X, dtype=float)
        n_train = self.X_train.shape[0]
        n_test = X.shape[0]

        d2 = self._pairwise_sq_dist(X, self.X_train)
        k = min(self.n_neighbors, n_train)
        idx = np.argpartition(d2, kth=max(1, k - 1), axis=1)[:, :k]

        cross = np.zeros((n_test, n_train), dtype=float)
        for i in range(n_test):
            nbrs = idx[i]
            cross[i, nbrs] = np.exp(-d2[i, nbrs])

        top = np.hstack([self.pruned_graph, cross.T])
        bottom = np.hstack([cross, np.zeros((n_test, n_test), dtype=float)])
        A_full = np.vstack([top, bottom])

        labels = np.concatenate([self.y_train, np.array([-1] * n_test)])
        Y = self._label_propagation(A_full, labels, n_train)
        probs = Y[-n_test:]
        probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("SpectralGraphPruningClassifier is not fitted.")
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class HistogramGradientBoosting:
    """Compact histogram-style gradient boosting for binary classification."""

    def __init__(self, n_estimators: int = 120, learning_rate: float = 0.1, max_depth: int = 4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees: List[dict] = []
        self.base_score: float = 0.0

    def _best_split(self, X: np.ndarray, gradients: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for f in range(X.shape[1]):
            values = np.unique(X[:, f])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for t in thresholds:
                left = X[:, f] <= t
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue
                g_left = gradients[left].sum()
                g_right = gradients[right].sum()
                gain = (g_left ** 2) / (left.sum() + 1e-12) + (g_right ** 2) / (right.sum() + 1e-12)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = float(t)

        return best_feature, best_threshold, float(best_gain)

    def _build_tree(self, X: np.ndarray, gradients: np.ndarray, depth: int = 0):
        if depth >= self.max_depth or len(X) < 8:
            return float(np.mean(gradients))

        f, t, gain = self._best_split(X, gradients)
        if f is None or gain <= 0:
            return float(np.mean(gradients))

        left = X[:, f] <= t
        return {
            "feature": f,
            "threshold": t,
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
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("HistogramGradientBoosting currently supports binary classification only.")

        y_bin = np.where(y == classes[0], 0.0, 1.0)
        p0 = np.clip(y_bin.mean(), 1e-6, 1 - 1e-6)
        self.base_score = float(np.log(p0 / (1 - p0)))
        raw = np.full(len(y_bin), self.base_score)

        self.trees = []
        for _ in range(self.n_estimators):
            prob = self._sigmoid(raw)
            gradients = y_bin - prob
            tree = self._build_tree(X, gradients, depth=0)
            self.trees.append(tree)
            update = np.array([self._predict_tree_one(x, tree) for x in X])
            raw += self.learning_rate * update

        self._classes = classes
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
        labels = np.where(probs[:, 1] >= 0.5, self._classes[1], self._classes[0])
        return labels

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


def create_synthetic_data(
    n_samples: int = 800,
    n_features: int = 6,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))

    if n_classes == 2:
        decision = X[:, 0] + 0.6 * X[:, 1] - 0.35 * X[:, 2] + 0.25 * rng.normal(size=n_samples)
        y = (decision > 0).astype(int)
    else:
        raw = X[:, : min(3, n_features)] @ np.arange(1, min(3, n_features) + 1)
        raw += 0.3 * rng.normal(size=n_samples)
        bins = np.quantile(raw, q=np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(raw, bins)

    return X, y


def create_regression_data(
    n_samples: int = 300,
    n_features: int = 3,
    noise_std: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    coef = np.linspace(1.0, 2.0, n_features)
    y = 1.5 + X @ coef + rng.normal(scale=noise_std, size=n_samples)
    return X, y


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_test = int(len(X) * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def benchmark() -> Dict[str, float]:
    X, y = create_synthetic_data(n_samples=900, n_features=6, n_classes=2, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

    models = {
        "DecisionTree": DecisionTree(max_depth=8),
        "KNearestNeighbors": KNearestNeighbors(k=7),
        "LinearSVM": LinearSVM(n_iterations=700, learning_rate=0.01),
        "SpectralGraphPruningClassifier": SpectralGraphPruningClassifier(
            n_neighbors=10,
            n_components=5,
            prune_threshold=0.25,
            n_iterations=20,
        ),
        "HistogramGradientBoosting": HistogramGradientBoosting(n_estimators=90, max_depth=4, learning_rate=0.1),
    }

    results: Dict[str, float] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = model.score(X_test, y_test)

    return results


def demo_regression() -> float:
    X, y = create_regression_data(n_samples=400, n_features=3, random_state=11)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=11)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)


if __name__ == "__main__":
    print("=" * 72)
    print("NON-NEURAL MACHINE LEARNING TOOLKIT DEMO")
    print("=" * 72)

    print("\nClassification benchmark on synthetic data:")
    scores = benchmark()
    for name, score in scores.items():
        print(f"{name:34s} accuracy = {score:.4f}")

    winner = max(scores, key=scores.get)
    print(f"\nWinner: {winner} ({scores[winner]:.4f})")

    print("\nRegression demo (LinearRegression):")
    mse = demo_regression()
    print(f"MSE = {mse:.4f}")
