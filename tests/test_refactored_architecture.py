import os
import tempfile
import unittest

from non_neuralx import (
    AgentConfig,
    HybridAgentV2,
    NonNeuralAgentV2,
    SpectralConfig,
    SpectralGraphClassifier,
    create_synthetic_data,
)


class TestRefactoredArchitecture(unittest.TestCase):
    def test_spectral_graph_classifier_with_config(self):
        X, y = create_synthetic_data(n_samples=120, n_features=6, n_classes=2, random_state=13)
        model = SpectralGraphClassifier(config=SpectralConfig(n_neighbors=8, n_components=3, random_state=13))
        model.fit(X[:90], y[:90])
        preds = model.predict(X[90:])
        self.assertEqual(len(preds), len(y[90:]))

    def test_non_neural_agent_v2_pipeline(self):
        corpus = (
            "Non-neural methods are interpretable. "
            "Spectral memory retrieves relevant context. "
            "Symbolic reasoning supports transparent inference. "
        ) * 20
        agent = NonNeuralAgentV2(config=AgentConfig(retrieval_top_k=3, reasoning_top_k=3, reasoning_max_steps=4))
        agent.learn(corpus)
        result = agent.think("why does symbolic reasoning help?")
        self.assertIn("answer", result)
        self.assertIn("reasoning_trace", result)
        self.assertIn("confidence", result)
        self.assertIn("route_confidence", result)
        self.assertIn("timings_ms", result)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertTrue(0.0 <= result["route_confidence"] <= 1.0)
        self.assertTrue(result["timings_ms"]["retrieval"] >= 0.0)
        self.assertTrue(result["timings_ms"]["reasoning"] >= 0.0)
        self.assertTrue(result["timings_ms"]["response"] >= 0.0)
        self.assertTrue(result["timings_ms"]["total"] >= 0.0)

    def test_spectral_graph_classifier_save_load(self):
        X, y = create_synthetic_data(n_samples=120, n_features=6, n_classes=2, random_state=19)
        model = SpectralGraphClassifier(config=SpectralConfig(n_neighbors=8, n_components=3, random_state=19))
        model.fit(X[:90], y[:90])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spectral.pkl")
            model.save(path)
            loaded = SpectralGraphClassifier.load(path)

        self.assertEqual(list(model.predict(X[90:])), list(loaded.predict(X[90:])))

    def test_non_neural_agent_v2_save_load(self):
        corpus = (
            "Non-neural methods are interpretable. "
            "Spectral memory retrieves relevant context. "
            "Symbolic reasoning supports transparent inference. "
        ) * 20
        agent = NonNeuralAgentV2()
        agent.learn(corpus)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "agent.pkl")
            agent.save(path)
            restored = NonNeuralAgentV2.load(path)

        result = restored.think("why does symbolic reasoning help?")
        self.assertIn("answer", result)
        self.assertIn("confidence", result)

    def test_hybrid_agent_v2_adapter(self):
        corpus = (
            "Symbolic reasoning supports transparent inference. "
            "Spectral memory retrieves grounded evidence. "
            "Markov generation follows stylistic context. "
        ) * 10
        agent = HybridAgentV2()
        agent.learn(corpus)
        result = agent.think("Explain why spectral memory retrieves evidence.")
        self.assertIn("answer", result)
        self.assertIn("route", result)


if __name__ == "__main__":
    unittest.main()
