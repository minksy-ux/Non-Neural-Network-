"""NonNeuralX - Pure non-neural machine learning toolkit."""

from .agents import HybridAgentV2, NonNeuralAgentV2
from .advanced_markov import RetrievalAugmentedMarkov
from .character_markov import CharacterMarkovGenerator
from .config import AgentConfig, HybridConfig, SpectralConfig
from .core import BaseClassifier, SimpleVotingEnsemble, SpectralGraphClassifier
from .data import create_synthetic_data
from .decision_tree import DecisionTree
from .histogram_boosting import HistogramGradientBoosting
from .hybrid_agent import NonNeuralAgent
from .knn import KNearestNeighbors
from .neural_symbolic_hybrid import (
    GraphSpectralMemory,
    InterpretableDecisionEnsemble,
    NetworkXSymbolicReasoner,
    NeuralSymbolicHybridAgent,
    TorchEmbeddingEncoder,
)
from .spectral_graph_pruning import SpectralGraphPruningClassifier
from .spectral_memory import SpectralMemory
from .symbolic_reasoner import SymbolicReasoner
from .utils import accuracy_score, mean_squared_error
from .visualization import plot_graph, plot_pruning_effect, plot_spectral_embedding

__version__ = "2.0.0"

__all__ = [
    "DecisionTree",
    "KNearestNeighbors",
    "SpectralGraphPruningClassifier",
    "HistogramGradientBoosting",
    "TorchEmbeddingEncoder",
    "NetworkXSymbolicReasoner",
    "GraphSpectralMemory",
    "InterpretableDecisionEnsemble",
    "NeuralSymbolicHybridAgent",
    "CharacterMarkovGenerator",
    "RetrievalAugmentedMarkov",
    "SpectralMemory",
    "SymbolicReasoner",
    "NonNeuralAgent",
    "create_synthetic_data",
    "accuracy_score",
    "mean_squared_error",
    "plot_spectral_embedding",
    "plot_pruning_effect",
    "plot_graph",
    "SpectralConfig",
    "AgentConfig",
    "HybridConfig",
    "BaseClassifier",
    "SimpleVotingEnsemble",
    "SpectralGraphClassifier",
    "NonNeuralAgentV2",
    "HybridAgentV2",
]
