from abc import ABC, abstractmethod
from typing import List

from ...spectral_memory import SpectralMemory


class BaseMemory(ABC):
    """Abstract retrieval memory interface."""

    @abstractmethod
    def add_text(self, text: str) -> None:
        """Ingest raw text into memory."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k memory chunks relevant to query."""


class SpectralMemoryComponent(BaseMemory):
    """Adapter exposing SpectralMemory under the BaseMemory interface."""

    def __init__(self, chunk_size: int = 120, vector_size: int = 512):
        self.impl = SpectralMemory(chunk_size=chunk_size, vector_size=vector_size)

    def add_text(self, text: str) -> None:
        self.impl.add_text(text)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        return self.impl.retrieve(query, top_k=top_k)
