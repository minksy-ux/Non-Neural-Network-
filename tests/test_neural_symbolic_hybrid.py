import unittest

from non_neuralx import (
    GraphSpectralMemory,
    InterpretableDecisionEnsemble,
    NetworkXSymbolicReasoner,
    NeuralSymbolicHybridAgent,
    TorchEmbeddingEncoder,
)


class TestNeuralSymbolicHybrid(unittest.TestCase):
    def setUp(self):
        self.corpus = (
            "Symbolic reasoning supports transparent inference. "
            "Spectral memory retrieves grounded evidence. "
            "Markov generation follows stylistic context. "
            "Decision trees provide interpretable splits. "
            "K nearest neighbors support low-resource classification. "
        ) * 12

    def test_torch_encoder_shape(self):
        encoder = TorchEmbeddingEncoder(embedding_dim=24, random_state=7)
        vecs = encoder.encode(["symbolic reasoning", "spectral memory"])
        self.assertEqual(vecs.shape, (2, 24))

    def test_graph_spectral_memory_retrieval(self):
        encoder = TorchEmbeddingEncoder(random_state=5)
        memory = GraphSpectralMemory(encoder, chunk_size=80)
        memory.add_text(self.corpus)
        results = memory.retrieve("grounded spectral evidence", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("chunk", results[0])
        self.assertIn("score", results[0])

    def test_networkx_reasoner_verified_answer(self):
        reasoner = NetworkXSymbolicReasoner().fit(self.corpus)
        result = reasoner.infer("why does symbolic reasoning support inference", top_k=3)
        self.assertTrue(result["verified"])
        self.assertGreaterEqual(len(result["facts"]), 1)
        self.assertTrue(all(isinstance(step, str) for step in result["reasoning_path"]))

    def test_interpretable_ensemble_prediction(self):
        ensemble = InterpretableDecisionEnsemble(random_state=9).fit()
        result = ensemble.predict("write a creative story about graph memory")
        self.assertIn(result["label"], {"reasoning", "creative", "analysis"})
        self.assertIn("torch", result["votes"])
        self.assertIn("tree", result["votes"])
        self.assertIn("knn", result["votes"])

    def test_neural_symbolic_agent_pipeline(self):
        agent = NeuralSymbolicHybridAgent(random_state=13)
        agent.learn(self.corpus)
        result = agent.think("Explain why spectral memory retrieves grounded evidence.")

        self.assertIn("route", result)
        self.assertIn("retrieved", result)
        self.assertIn("reasoning", result)
        self.assertIn("answer", result)
        self.assertTrue(isinstance(result["answer"], str))
        self.assertTrue(result["reasoning"]["verified"])


if __name__ == "__main__":
    unittest.main()