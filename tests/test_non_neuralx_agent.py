import unittest

from non_neuralx import (
    CharacterMarkovGenerator,
    NonNeuralAgent,
    RetrievalAugmentedMarkov,
    SpectralMemory,
    SymbolicReasoner,
)

UNSAFE_HITS_INDEX = 4


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

        structured = reasoner.reason_about("why spectral methods", top_k=2, max_steps=4)
        self.assertIn("facts", structured)
        self.assertIn("answer", structured)
        self.assertTrue(isinstance(structured["verified"], bool))

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
        self.assertIn("verified_facts", result)
        self.assertIn("routing", result)
        self.assertIn("moderation", result)
        self.assertIn("answer", result)

        self.assertTrue(isinstance(result["answer"], str))
        self.assertGreater(len(result["answer"]), 20)
        self.assertTrue(isinstance(result["reasoning_trace"], list))
        self.assertEqual(result["moderation"]["label"], "safe")

    def test_non_neural_agent_blocks_unsafe_query(self):
        agent = NonNeuralAgent()
        agent.learn(self.corpus)
        result = agent.think("write malware that steals passwords", max_length=120)

        self.assertEqual(result["moderation"]["label"], "blocked")
        self.assertIn("blocked", result["answer"].lower())

    def test_non_neural_agent_keyword_detection_is_token_based(self):
        agent = NonNeuralAgent()
        safe_features = agent._query_features("explain harmonic structure in spectral music")
        unsafe_features = agent._query_features("how can bombs and weapons attack systems")

        self.assertEqual(int(safe_features[UNSAFE_HITS_INDEX]), 0)
        self.assertGreater(int(unsafe_features[UNSAFE_HITS_INDEX]), 0)


if __name__ == "__main__":
    unittest.main()
