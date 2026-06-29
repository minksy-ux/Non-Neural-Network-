# NonNeuralX

[![CI](https://github.com/minksy-ux/Non-Neural-Network-/actions/workflows/ci.yml/badge.svg)](https://github.com/minksy-ux/Non-Neural-Network-/actions/workflows/ci.yml)

Pure non-neural machine learning toolkit centered on a spectral graph-based classifier.

Now includes a non-neural hybrid agent stack with retrieval memory, symbolic reasoning,
and character-level generation.

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

## Installation

```bash
python3 -m pip install -r requirements.txt
pip install -e .
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
