from .character_markov import CharacterMarkovGenerator
from .spectral_memory import SpectralMemory


class RetrievalAugmentedMarkov(CharacterMarkovGenerator):
    """Character Markov generator with retrieval-augmented prompts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory = SpectralMemory()

    def learn(self, corpus: str, verbose: bool = True) -> "RetrievalAugmentedMarkov":
        self.memory.add_text(corpus)
        super().fit(corpus, verbose=verbose)
        return self

    def generate(self, prompt: str = "", max_length: int = 600, top_k: int = 3, stop_at_punctuation: bool = True) -> str:
        retrieved = self.memory.retrieve(prompt, top_k=top_k)
        context = "\n".join(retrieved)
        enhanced_prompt = f"Context: {context[:400]}\nPrompt: {prompt}\n"
        return super().generate(
            prompt=enhanced_prompt,
            max_length=max_length,
            stop_at_punctuation=stop_at_punctuation,
        )
