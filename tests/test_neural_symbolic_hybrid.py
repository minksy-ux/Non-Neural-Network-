import glob
import json
import os
import subprocess
import sys
import tempfile
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

    def test_torch_encoder_empty_batch(self):
        encoder = TorchEmbeddingEncoder(embedding_dim=16, random_state=7)
        vecs = encoder.encode([])
        self.assertEqual(vecs.shape, (0, 16))

    def test_torch_encoder_indices_stable_across_hash_seeds(self):
        script = (
            "import json; "
            "from non_neuralx.neural_symbolic_hybrid import TorchEmbeddingEncoder; "
            "enc=TorchEmbeddingEncoder(vocab_size=4096, embedding_dim=8, random_state=11); "
            "print(json.dumps(enc._indices('symbolic reasoning').tolist()))"
        )
        env_a = dict(os.environ, PYTHONHASHSEED="1")
        env_b = dict(os.environ, PYTHONHASHSEED="999")
        out_a = subprocess.check_output([sys.executable, "-c", script], env=env_a, text=True).strip()
        out_b = subprocess.check_output([sys.executable, "-c", script], env=env_b, text=True).strip()
        self.assertEqual(json.loads(out_a), json.loads(out_b))

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

    def test_networkx_reasoner_sympy_optional_consistency(self):
        reasoner = NetworkXSymbolicReasoner(enable_sympy_checks=True)
        reasoner.fit("system is safe. system is not safe.")
        result = reasoner.infer("system", top_k=4)
        self.assertIn("contradictions", result)
        self.assertGreaterEqual(len(result["contradictions"]), 1)

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

    def test_torch_encoder_disk_cache_optional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = TorchEmbeddingEncoder(embedding_dim=8, random_state=5, cache_dir=tmpdir, use_disk_cache=True)
            vecs = encoder.encode(["symbolic cache test", "symbolic cache test"])
            self.assertEqual(vecs.shape, (2, 8))
            files = glob.glob(os.path.join(tmpdir, "embedding_*.joblib"))
            # If joblib is unavailable this remains optional and no files are required.
            if encoder.use_disk_cache:
                self.assertGreaterEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main()