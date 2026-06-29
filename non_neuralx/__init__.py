"""NonNeuralX - Pure non-neural machine learning toolkit."""

from .advanced_markov import RetrievalAugmentedMarkov
from .character_markov import CharacterMarkovGenerator
from .decision_tree import DecisionTree
from .histogram_boosting import HistogramGradientBoosting
from .hybrid_agent import NonNeuralAgent
from .knn import KNearestNeighbors
from .spectral_graph_pruning import SpectralGraphPruningClassifier
from .spectral_memory import SpectralMemory
from .symbolic_reasoner import SymbolicReasoner
from .utils import accuracy_score, create_synthetic_data, mean_squared_error
from .visualization import plot_graph, plot_pruning_effect, plot_spectral_embedding

__version__ = "2.0.0"

__all__ = [
    "DecisionTree",
    "KNearestNeighbors",
    "SpectralGraphPruningClassifier",
    "HistogramGradientBoosting",
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
]
