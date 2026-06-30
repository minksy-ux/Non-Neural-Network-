from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ...symbolic_reasoner import SymbolicReasoner


class BaseReasoner(ABC):
    """Abstract symbolic reasoner interface."""

    @abstractmethod
    def fit(self, corpus: str) -> None:
        """Build reasoning structures from corpus."""

    @abstractmethod
    def reason_about(self, query: str, top_k: int = 4, max_steps: int = 5) -> Dict[str, Any]:
        """Return structured reasoning output for query."""


class SymbolicReasonerComponent(BaseReasoner):
    """Adapter exposing SymbolicReasoner under the BaseReasoner interface."""

    def __init__(self):
        self.impl = SymbolicReasoner()

    def fit(self, corpus: str) -> None:
        self.impl.fit(corpus)

    def reason_about(self, query: str, top_k: int = 4, max_steps: int = 5) -> Dict[str, Any]:
        return self.impl.reason_about(query, top_k=top_k, max_steps=max_steps)

    def reason(self, query: str, max_steps: int = 5) -> List[str]:
        return self.impl.reason(query, max_steps=max_steps)
