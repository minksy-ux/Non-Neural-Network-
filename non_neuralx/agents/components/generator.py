from abc import ABC, abstractmethod

from ...character_markov import CharacterMarkovGenerator


class BaseGenerator(ABC):
    """Abstract text generation interface."""

    @abstractmethod
    def fit(self, corpus: str) -> None:
        """Train generator from corpus."""

    @abstractmethod
    def generate(self, prompt: str, max_length: int = 450) -> str:
        """Generate text from prompt."""


class CharacterMarkovComponent(BaseGenerator):
    """Adapter exposing CharacterMarkovGenerator under BaseGenerator."""

    def __init__(self, order: int = 10, temperature: float = 0.95, backoff: bool = True, random_state: int = 42):
        self.impl = CharacterMarkovGenerator(order=order, temperature=temperature, backoff=backoff, random_state=random_state)

    def fit(self, corpus: str) -> None:
        self.impl.fit(corpus, verbose=False)

    def generate(self, prompt: str, max_length: int = 450) -> str:
        return self.impl.generate(prompt=prompt, max_length=max_length)
