from .advanced_markov import RetrievalAugmentedMarkov
from .spectral_graph_pruning import SpectralGraphPruningClassifier
from .spectral_memory import SpectralMemory
from .symbolic_reasoner import SymbolicReasoner


class NonNeuralAgent:
    """Non-neural hybrid agent with retrieval, symbolic reasoning, and generation."""

    def __init__(self):
        self.memory = SpectralMemory()
        self.markov = RetrievalAugmentedMarkov(order=12, temperature=0.88, backoff=True)
        self.reasoner = SymbolicReasoner()
        self.classifier = SpectralGraphPruningClassifier()

    def learn(self, corpus: str) -> None:
        self.memory.add_text(corpus)
        self.markov.learn(corpus, verbose=False)
        self.reasoner.fit(corpus)

    def think(self, query: str, max_length: int = 450) -> dict:
        retrieved = self.memory.retrieve(query, top_k=4)
        reasoning = self.reasoner.reason(query)
        answer = self.markov.generate(prompt=query, max_length=max_length, top_k=4)
        return {
            "query": query,
            "retrieved_context": retrieved,
            "reasoning_trace": reasoning,
            "answer": answer,
        }
