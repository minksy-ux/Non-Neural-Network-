from abc import ABC, abstractmethod
from typing import Tuple


class BaseRouter(ABC):
    """Abstract route decision interface for agent pipelines."""

    @abstractmethod
    def route_with_confidence(self, query: str) -> Tuple[str, float]:
        """Return route label and confidence in [0, 1]."""

    def route(self, query: str) -> str:
        return self.route_with_confidence(query)[0]


class KeywordRouter(BaseRouter):
    """Simple fallback router based on query keyword heuristics."""

    def route_with_confidence(self, query: str) -> Tuple[str, float]:
        lower = query.lower()
        code_hits = sum(word in lower for word in ["code", "debug", "function", "test"])
        reasoning_hits = sum(word in lower for word in ["why", "how", "explain", "logic"])
        creative_hits = sum(word in lower for word in ["write", "poem", "story", "creative", "style"])

        if code_hits >= max(reasoning_hits, creative_hits) and code_hits > 0:
            return "code", min(1.0, 0.45 + 0.2 * code_hits)
        if reasoning_hits >= max(code_hits, creative_hits) and reasoning_hits > 0:
            return "reasoning", min(1.0, 0.45 + 0.2 * reasoning_hits)
        if creative_hits > 0:
            return "creative", min(1.0, 0.45 + 0.2 * creative_hits)
        return "creative", 0.4

    def route(self, query: str) -> str:
        return self.route_with_confidence(query)[0]
