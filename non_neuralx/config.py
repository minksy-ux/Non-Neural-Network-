from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SpectralConfig:
    """Configuration for spectral graph classifiers."""

    n_neighbors: int = 12
    n_components: int = 6
    prune_threshold: float = 0.25
    n_iterations: int = 25
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if self.prune_threshold < 0.0:
            raise ValueError("prune_threshold must be >= 0")


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for non-neural agent pipeline behavior."""

    retrieval_top_k: int = 4
    reasoning_top_k: int = 4
    reasoning_max_steps: int = 5
    max_generation_length: int = 450

    def __post_init__(self) -> None:
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be >= 1")
        if self.reasoning_top_k < 1:
            raise ValueError("reasoning_top_k must be >= 1")
        if self.reasoning_max_steps < 1:
            raise ValueError("reasoning_max_steps must be >= 1")
        if self.max_generation_length < 1:
            raise ValueError("max_generation_length must be >= 1")


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for hybrid agent mode and optional components."""

    random_state: int = 42
    max_tokens: int = 80
    cache_dir: Optional[str] = None
    use_disk_cache: bool = False
    enable_sympy_checks: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
