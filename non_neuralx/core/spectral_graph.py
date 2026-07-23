import pickle
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..config import SpectralConfig
from ..spectral_graph_pruning import SpectralGraphPruningClassifier
from .base import BaseClassifier


class SpectralGraphClassifier(SpectralGraphPruningClassifier, BaseClassifier):
    """Typed wrapper around SpectralGraphPruningClassifier using dataclass config."""

    def __init__(self, config: Optional[SpectralConfig] = None):
        self.config = config or SpectralConfig()
        super().__init__(
            n_neighbors=self.config.n_neighbors,
            n_components=self.config.n_components,
            prune_threshold=self.config.prune_threshold,
            n_iterations=self.config.n_iterations,
            random_state=self.config.random_state,
        )

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int_]) -> "SpectralGraphClassifier":
        super().fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return self

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as handle:
            pickle.dump({"config": self.config, "state": self.__dict__}, handle)

    @classmethod
    def load(cls, filepath: str) -> "SpectralGraphClassifier":
        with open(filepath, "rb") as handle:
            payload = pickle.load(handle)
        model = cls(config=payload["config"])
        model.__dict__.update(payload["state"])
        return model
