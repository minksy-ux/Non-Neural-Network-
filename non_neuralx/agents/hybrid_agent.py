import pickle
from typing import Any, Dict

from ..config import HybridConfig
from ..neural_symbolic_hybrid import NeuralSymbolicHybridAgent
from .base_agent import BaseAgent


class HybridAgentV2(BaseAgent):
    """Adapter-based hybrid agent entry point with centralized config."""

    def __init__(self, config: HybridConfig | None = None):
        self.config = config or HybridConfig()
        self.impl = NeuralSymbolicHybridAgent(
            random_state=self.config.random_state,
            cache_dir=self.config.cache_dir,
            use_disk_cache=self.config.use_disk_cache,
            enable_sympy_checks=self.config.enable_sympy_checks,
        )

    def learn(self, corpus: str) -> None:
        self.impl.learn(corpus)

    def think(self, query: str) -> Dict[str, Any]:
        return self.impl.think(query, max_tokens=self.config.max_tokens)

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, filepath: str) -> "HybridAgentV2":
        with open(filepath, "rb") as handle:
            agent = pickle.load(handle)
        if not isinstance(agent, cls):
            raise TypeError("Persisted object is not a HybridAgentV2 instance")
        return agent
