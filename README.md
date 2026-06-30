# NonNeuralX

[![CI](https://github.com/minksy-ux/Non-Neural-Network-/actions/workflows/ci.yml/badge.svg)](https://github.com/minksy-ux/Non-Neural-Network-/actions/workflows/ci.yml)

Pure non-neural machine learning toolkit centered on a spectral graph-based classifier.

Now includes a non-neural hybrid agent stack with retrieval memory, symbolic reasoning,
and character-level generation.

An opt-in neural-symbolic hybrid is also available for users who want torch embeddings
plus graph-based symbolic control while keeping the reasoning path inspectable.

## Project Structure

```
NonNeuralX/
├── non_neuralx/
│   ├── __init__.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── spectral_graph_pruning.py
│   ├── histogram_boosting.py
│   ├── character_markov.py
│   ├── spectral_memory.py
│   ├── symbolic_reasoner.py
│   ├── advanced_markov.py
│   ├── hybrid_agent.py
│   ├── utils.py
│   └── visualization.py
├── benchmark.py
├── example.py
├── requirements.txt
└── pyproject.toml
```

## Models

- SpectralGraphPruningClassifier (flagship graph-spectral model)
- HistogramGradientBoosting
- DecisionTree
- KNearestNeighbors

## Agent Components

- CharacterMarkovGenerator
- RetrievalAugmentedMarkov
- SpectralMemory
- SymbolicReasoner
- NonNeuralAgent

## Optional Neural-Symbolic Hybrid

The package now also includes a lightweight hybrid stack that uses:

- `TorchEmbeddingEncoder` for neural input embeddings
- `GraphSpectralMemory` for NetworkX Laplacian retrieval
- `NetworkXSymbolicReasoner` for verified graph reasoning and consistency checks
- `InterpretableDecisionEnsemble` for torch + manual tree + k-NN routing
- `NeuralSymbolicHybridAgent` for end-to-end orchestration

```python
from non_neuralx import NeuralSymbolicHybridAgent

corpus = (
	"Symbolic reasoning supports transparent inference. "
	"Spectral memory retrieves grounded evidence. "
	"Markov generation follows stylistic context. "
) * 20

agent = NeuralSymbolicHybridAgent()
agent.learn(corpus)

result = agent.think("Explain why symbolic reasoning supports transparent inference.")
print(result["route"])
print(result["answer"])
```

## Installation

```bash
python3 -m pip install -r requirements.txt
pip install -e .
```

## Contributing

Run this from a fresh clone to set up and execute tests in one command:

```bash
python3 -m pip install -r requirements.txt && pip install -e . && PYTHONDONTWRITEBYTECODE=1 pytest -q
```

## Quick Start

```python
from sklearn.model_selection import train_test_split
from non_neuralx import SpectralGraphPruningClassifier, create_synthetic_data

X, y = create_synthetic_data(n_samples=1500, n_features=8, n_classes=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SpectralGraphPruningClassifier(n_neighbors=12, n_components=6)
model.fit(X_train, y_train)
print("SGPC Accuracy:", model.score(X_test, y_test))
```

## Agent Quick Start

```python
from non_neuralx import NonNeuralAgent

corpus = (
	"Non-neural systems are efficient and interpretable. "
	"Spectral methods reveal structure in data. "
	"Symbolic reasoning supports transparent inference. "
) * 40

agent = NonNeuralAgent()
agent.learn(corpus)

result = agent.think("Why are non-neural approaches useful?")
print(result["answer"])
print(result["reasoning_trace"])
```

## Benchmark

```bash
python3 benchmark.py
```

## Existing Monolithic Module

The original consolidated implementation remains available in `non_neural_toolkit.py`.
