from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int_]


class BaseClassifier(ABC):
    """Abstract classifier contract for non-neural models."""

    @abstractmethod
    def fit(self, X: ArrayF, y: ArrayI) -> "BaseClassifier":
        """Train the classifier on feature matrix X and labels y."""

    @abstractmethod
    def predict(self, X: ArrayF) -> ArrayI:
        """Predict class labels for feature matrix X."""

    @abstractmethod
    def score(self, X: ArrayF, y: ArrayI) -> float:
        """Return scalar model score against labels y."""

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist model state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseClassifier":
        """Restore model state from disk."""
