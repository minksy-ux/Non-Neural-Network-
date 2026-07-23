"""Core non-neural model abstractions and wrappers."""

from .base import BaseClassifier
from .ensembles import SimpleVotingEnsemble
from .spectral_graph import SpectralGraphClassifier

__all__ = ["BaseClassifier", "SimpleVotingEnsemble", "SpectralGraphClassifier"]
