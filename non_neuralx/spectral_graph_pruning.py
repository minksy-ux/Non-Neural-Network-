from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


class SpectralGraphPruningClassifier:
    """
    Spectral Graph Pruning Classifier (SGPC).

    Pipeline:
    - Build weighted k-NN graph
    - Compute normalized Laplacian eigenvectors
    - Prune edges with low spectral coherence
    - Predict via label propagation on train+test graph
    """

    def __init__(
        self,
        n_neighbors: int = 12,
        n_components: int = 6,
        prune_threshold: float = 0.25,
        n_iterations: int = 25,
        random_state: int = 42,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.prune_threshold = prune_threshold
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.embedding: Optional[np.ndarray] = None
        self.pruned_graph: Optional[sp.csr_matrix] = None
        self.classes_: Optional[np.ndarray] = None

    def _build_knn_graph(self, X: np.ndarray) -> sp.csr_matrix:
        n = X.shape[0]
        k = min(self.n_neighbors, max(1, n - 1))
        dist = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        np.fill_diagonal(dist, np.inf)

        idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        row = np.repeat(np.arange(n), k)
        col = idx.ravel()
        data = np.exp(-dist[row, col])

        adj = sp.csr_matrix((data, (row, col)), shape=(n, n))
        return ((adj + adj.T) * 0.5).tocsr()

    def _prune_graph(self, adj: sp.csr_matrix, emb: np.ndarray) -> sp.csr_matrix:
        row, col = adj.nonzero()
        coherence = np.sum((emb[row] - emb[col]) ** 2, axis=1)

        threshold = self.prune_threshold * (np.mean(coherence) + 0.5 * np.std(coherence))
        mask = coherence < threshold

        pruned = sp.csr_matrix((adj.data[mask], (row[mask], col[mask])), shape=adj.shape)
        return pruned.maximum(pruned.T).tocsr()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralGraphPruningClassifier":
        np.random.seed(self.random_state)
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=int)
        self.classes_ = np.unique(self.y_train)

        adj = self._build_knn_graph(self.X_train)

        degree = np.asarray(adj.sum(1)).ravel() + 1e-8
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(degree))
        laplacian = sp.eye(adj.shape[0], format="csr") - d_inv_sqrt @ adj @ d_inv_sqrt

        components = min(self.n_components + 1, max(2, adj.shape[0] - 1))
        try:
            _, vecs = eigsh(laplacian, k=components, which="SM", tol=1e-6)
            self.embedding = vecs[:, 1:]
        except Exception:
            # Fallback for degenerate graphs.
            self.embedding = np.random.randn(self.X_train.shape[0], max(1, self.n_components))

        self.pruned_graph = self._prune_graph(adj, self.embedding)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None or self.pruned_graph is None or self.classes_ is None:
            raise RuntimeError("SpectralGraphPruningClassifier is not fitted.")

        X = np.asarray(X, dtype=float)
        n_test = X.shape[0]
        n_train = self.X_train.shape[0]

        full_X = np.vstack([self.X_train, X])
        adj_test = self._build_knn_graph(full_X)[n_train:, :n_train].tocsr()

        full_adj = sp.block_diag((self.pruned_graph, sp.csr_matrix((n_test, n_test))), format="lil")
        full_adj[n_train:, :n_train] = adj_test
        full_adj[:n_train, n_train:] = adj_test.T
        full_adj = full_adj.tocsr()

        Y = np.zeros((n_train + n_test, len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            train_idx = np.where(self.y_train == cls)[0]
            Y[train_idx, i] = 1.0

        train_hot = np.eye(len(self.classes_))[np.searchsorted(self.classes_, self.y_train)]

        for _ in range(self.n_iterations):
            Y = full_adj @ Y
            Y /= np.maximum(Y.sum(axis=1, keepdims=True), 1e-10)
            Y[:n_train] = train_hot

        return Y[-n_test:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("SpectralGraphPruningClassifier is not fitted.")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
