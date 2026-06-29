import re

import numpy as np


class SpectralMemory:
    """Simple retrieval memory based on character-hash vectors."""

    def __init__(self, chunk_size: int = 120, vector_size: int = 512):
        self.chunk_size = chunk_size
        self.vector_size = vector_size
        self.chunks = []
        self.embeddings = None

    def add_text(self, text: str) -> None:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return

        stride = max(1, self.chunk_size // 2)
        windows = [text[i : i + self.chunk_size] for i in range(0, len(text), stride)]
        self.chunks.extend([c for c in windows if len(c) > 20])
        self._build_embeddings()

    def _chunk_vector(self, chunk: str) -> np.ndarray:
        vec = np.zeros(self.vector_size, dtype=float)
        for ch in chunk:
            vec[ord(ch) % self.vector_size] += 1.0
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _build_embeddings(self) -> None:
        if not self.chunks:
            self.embeddings = None
            return
        self.embeddings = np.vstack([self._chunk_vector(chunk) for chunk in self.chunks])

    def retrieve(self, query: str, top_k: int = 3):
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        qvec = self._chunk_vector(query)
        sims = self.embeddings @ qvec
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [self.chunks[i] for i in top_idx]
