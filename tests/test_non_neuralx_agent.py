import unittest

from non_neuralx import (
    CharacterMarkovGenerator,
    NonNeuralAgent,
    RetrievalAugmentedMarkov,
    SpectralMemory,
    SymbolicReasoner,
)


class TestNonNeuralXAgentModules(unittest.TestCase):
    def setUp(self):
        self.corpus = (
            "Non-neural methods are interpretable and efficient. "
            "Spectral analysis can reveal hidden structure in data. "
            "Symbolic reasoning supports transparent inference traces. "
        ) * 25

    def test_character_markov_fit_generate(self):
        model = CharacterMarkovGenerator(order=8, temperature=0.95, random_state=7)
        model.fit(self.corpus, verbose=False)
        out = model.generate("non-neural", max_length=120)
        self.assertTrue(isinstance(out, str))
        self.assertGreater(len(out), 20)

    def test_spectral_memory_retrieve(self):
        memory = SpectralMemory(chunk_size=80)
        memory.add_text(self.corpus)
        retrieved = memory.retrieve("spectral structure", top_k=3)
        self.assertGreaterEqual(len(retrieved), 1)
        self.assertLessEqual(len(retrieved), 3)

    def test_symbolic_reasoner_trace(self):
        reasoner = SymbolicReasoner().fit(self.corpus)
        trace = reasoner.reason("why spectral methods", max_steps=4)
        self.assertGreaterEqual(len(trace), 1)
        self.assertTrue(all(isinstance(step, str) for step in trace))

    def test_retrieval_augmented_markov(self):
        model = RetrievalAugmentedMarkov(order=8, temperature=0.95, random_state=11)
        model.learn(self.corpus, verbose=False)
        out = model.generate("spectral", max_length=100, top_k=2)
        self.assertGreater(len(out), 15)

    def test_non_neural_agent_pipeline(self):
        agent = NonNeuralAgent()
        agent.learn(self.corpus)
        result = agent.think("why use non-neural methods?", max_length=120)

        self.assertIn("query", result)
        self.assertIn("retrieved_context", result)
        self.assertIn("reasoning_trace", result)
        self.assertIn("answer", result)

        self.assertTrue(isinstance(result["answer"], str))
        self.assertGreater(len(result["answer"]), 20)
        self.assertTrue(isinstance(result["reasoning_trace"], list))


if __name__ == "__main__":
    unittest.main()
