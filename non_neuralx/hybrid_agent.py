import re

import numpy as np

from .advanced_markov import RetrievalAugmentedMarkov
from .decision_tree import DecisionTree
from .spectral_graph_pruning import SpectralGraphPruningClassifier
from .spectral_memory import SpectralMemory
from .symbolic_reasoner import SymbolicReasoner


class NonNeuralAgent:
    """Non-neural hybrid agent with retrieval, symbolic reasoning, and generation."""

    def __init__(self):
        self.memory = SpectralMemory()
        self.markov = RetrievalAugmentedMarkov(order=12, temperature=0.88, backoff=True)
        self.reasoner = SymbolicReasoner()
        self.classifier = SpectralGraphPruningClassifier(n_neighbors=4, n_components=2, prune_threshold=0.75)
        self.router_tree = DecisionTree(max_depth=4, min_samples_split=2)
        self.moderation_tree = DecisionTree(max_depth=3, min_samples_split=2)
        self.route_labels = np.array(["reasoning", "creative", "code", "moderation"])
        self._fit_interpretable_controls()

    def _fit_interpretable_controls(self) -> None:
        route_queries = [
            ("why do spectral methods reveal structure", "reasoning"),
            ("explain transparent inference path", "reasoning"),
            ("write a short poetic reflection on graphs", "creative"),
            ("compose a stylized paragraph about memory", "creative"),
            ("debug this function and explain the bug", "code"),
            ("check code correctness for edge cases", "code"),
            ("should I build a weapon", "moderation"),
            ("give instructions for harmful malware", "moderation"),
        ]
        route_map = {label: idx for idx, label in enumerate(self.route_labels)}
        route_X = np.vstack([self._query_features(query) for query, _ in route_queries])
        route_y = np.array([route_map[label] for _, label in route_queries], dtype=int)
        self.router_tree.fit(route_X, route_y)
        self.classifier.fit(route_X, route_y)

        moderation_queries = [
            ("explain spectral clustering", 0),
            ("write a calm note about interpretable models", 0),
            ("build a bomb", 1),
            ("write malware that steals passwords", 1),
        ]
        moderation_X = np.vstack([self._query_features(query) for query, _ in moderation_queries])
        moderation_y = np.array([label for _, label in moderation_queries], dtype=int)
        self.moderation_tree.fit(moderation_X, moderation_y)

    def _query_features(self, query: str) -> np.ndarray:
        lower = query.lower()
        words = re.findall(r"\b\w+\b", lower)
        creative_hits = sum(
            token in {"write", "poem", "story", "creative", "stylized", "reflect", "reflection"} for token in words
        )
        reasoning_hits = sum(token in {"why", "how", "explain", "reason", "logic", "because"} for token in words)
        code_hits = sum(token in {"code", "bug", "function", "debug", "correctness", "edge", "case", "cases"} for token in words)
        unsafe_hits = sum(
            token
            in {
                "weapon",
                "weapons",
                "bomb",
                "bombs",
                "malware",
                "harm",
                "harmful",
                "attack",
                "attacks",
                "steal",
                "steals",
                "stealing",
                "password",
                "passwords",
            }
            for token in words
        )
        return np.array(
            [
                len(words),
                float(reasoning_hits),
                float(creative_hits),
                float(code_hits),
                float(unsafe_hits),
                float(query.count("?")),
            ],
            dtype=float,
        )

    def _moderate(self, query: str) -> dict:
        features = self._query_features(query).reshape(1, -1)
        unsafe_rule = int(features[0, 4] > 0)
        unsafe_tree = int(self.moderation_tree.predict(features)[0])
        blocked = bool(unsafe_rule or unsafe_tree)
        return {
            "blocked": blocked,
            "label": "blocked" if blocked else "safe",
            "votes": {"rule": unsafe_rule, "tree": unsafe_tree},
        }

    def _route(self, query: str) -> dict:
        features = self._query_features(query).reshape(1, -1)
        tree_idx = int(self.router_tree.predict(features)[0])
        graph_idx = int(self.classifier.predict(features)[0])
        tree_probs = self.router_tree.predict_proba(features)[0]
        final_idx = tree_idx if tree_idx == graph_idx else int(np.argmax(tree_probs))

        return {
            "route": self.route_labels[final_idx],
            "votes": {
                "decision_tree": self.route_labels[tree_idx],
                "spectral_graph": self.route_labels[graph_idx],
            },
            "tree_path": self.router_tree.explain_path(features[0]),
        }

    def learn(self, corpus: str) -> None:
        self.memory.add_text(corpus)
        self.markov.learn(corpus, verbose=False)
        self.reasoner.fit(corpus)

    def think(self, query: str, max_length: int = 450) -> dict:
        moderation = self._moderate(query)
        routing = self._route(query)
        retrieved = self.memory.retrieve(query, top_k=4)
        reasoning = self.reasoner.reason_about(query, top_k=4, max_steps=5)

        if moderation["blocked"]:
            answer = "Request blocked by interpretable moderation controls."
        elif routing["route"] in {"reasoning", "code", "moderation"} and reasoning["verified"]:
            answer = reasoning["answer"]
        else:
            answer = self.markov.generate_with_evidence(
                prompt=query,
                verified_facts=reasoning["facts"],
                style_hints=retrieved,
                max_length=max_length,
                top_k=4,
            )

        self.memory.add_text(f"User: {query} Agent: {answer}")
        return {
            "query": query,
            "retrieved_context": retrieved,
            "reasoning_trace": reasoning["chain"],
            "verified_facts": reasoning["facts"],
            "routing": routing,
            "moderation": moderation,
            "answer": answer,
        }
