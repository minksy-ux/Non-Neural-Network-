from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """Abstract agent contract for learn/think pipelines."""

    @abstractmethod
    def learn(self, corpus: str) -> None:
        """Train internal components from text corpus."""

    @abstractmethod
    def think(self, query: str) -> Dict[str, Any]:
        """Produce an answer and trace for query."""

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist agent state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "BaseAgent":
        """Restore agent state from disk."""
