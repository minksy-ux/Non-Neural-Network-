import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

from .knn import KNearestNeighbors


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9_'-]+\b", _normalize_text(text).lower())


@dataclass
class VerifiedFact:
    subject: str
    relation: str
    obj: str
    sentence: str
    confidence: float = 1.0

    @property
    def content(self) -> str:
        return f"{self.subject} {self.relation} {self.obj}"


class TorchEmbeddingEncoder:
    """Small torch embedding-bag encoder for lightweight text representations."""

    def __init__(self, vocab_size: int = 4096, embedding_dim: int = 32, random_state: int = 42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.model = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        nn.init.xavier_uniform_(self.model.weight)
        self.model.eval()

    def _indices(self, text: str) -> torch.Tensor:
        tokens = _tokenize(text)
        if not tokens:
            return torch.tensor([0], dtype=torch.long)
        return torch.tensor([hash(token) % self.vocab_size for token in tokens], dtype=torch.long)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        with torch.no_grad():
            for text in texts:
                indices = self._indices(text)
                offsets = torch.tensor([0], dtype=torch.long)
                vec = self.model(indices, offsets).squeeze(0).cpu().numpy()
                norm = np.linalg.norm(vec) + 1e-8
                vectors.append(vec / norm)
        return np.vstack(vectors)


class NetworkXSymbolicReasoner:
    """Rule-oriented graph reasoner with contradiction checks over a NetworkX graph."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.facts: List[VerifiedFact] = []

    def fit(self, text: str) -> "NetworkXSymbolicReasoner":
        sentences = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
        for sentence in sentences:
            self._add_sentence(sentence)
        return self

    def _add_sentence(self, sentence: str) -> None:
        tokens = _tokenize(sentence)
        for left, right in zip(tokens, tokens[1:]):
            weight = self.graph[left][right]["weight"] + 1 if self.graph.has_edge(left, right) else 1
            self.graph.add_edge(left, right, relation="next", weight=weight)

        fact = self._extract_fact(sentence)
        if fact is None:
            return

        self.facts.append(fact)
        self.graph.add_node(fact.subject, kind="entity")
        self.graph.add_node(fact.obj, kind="entity")
        weight = self.graph[fact.subject][fact.obj]["weight"] + 1 if self.graph.has_edge(fact.subject, fact.obj) else 1
        self.graph.add_edge(fact.subject, fact.obj, relation=fact.relation, weight=weight, sentence=fact.sentence)

    def _extract_fact(self, sentence: str) -> Optional[VerifiedFact]:
        patterns = [
            ("is", r"^(.+?)\s+is\s+(?:an?|the)?\s*(.+)$"),
            ("are", r"^(.+?)\s+are\s+(?:an?|the)?\s*(.+)$"),
            ("retrieves", r"^(.+?)\s+retrieves\s+(.+)$"),
            ("uses", r"^(.+?)\s+uses\s+(.+)$"),
            ("supports", r"^(.+?)\s+supports\s+(.+)$"),
            ("prevents", r"^(.+?)\s+prevents\s+(.+)$"),
            ("reveals", r"^(.+?)\s+reveals?\s+(.+)$"),
        ]

        for relation, pattern in patterns:
            match = re.match(pattern, sentence, flags=re.IGNORECASE)
            if not match:
                continue
            subject = _normalize_text(match.group(1).lower())
            obj = _normalize_text(match.group(2).lower())
            if len(subject) < 2 or len(obj) < 2:
                return None
            return VerifiedFact(subject=subject, relation=relation, obj=obj, sentence=sentence)
        return None

    def retrieve_facts(self, query: str, top_k: int = 4) -> List[VerifiedFact]:
        query_tokens = set(_tokenize(query))
        scored: List[Tuple[float, VerifiedFact]] = []
        for fact in self.facts:
            fact_tokens = set(_tokenize(fact.content)) | set(_tokenize(fact.sentence))
            overlap = len(query_tokens & fact_tokens)
            if overlap == 0:
                continue
            scored.append((overlap + fact.confidence, fact))
        scored.sort(key=lambda item: (-item[0], item[1].sentence))
        return [fact for _, fact in scored[:top_k]]

    def infer(self, query: str, max_hops: int = 4, top_k: int = 4) -> Dict[str, object]:
        facts = self.retrieve_facts(query, top_k=top_k)
        query_tokens = _tokenize(query)
        path = []
        if query_tokens:
            current = query_tokens[0]
            for _ in range(max_hops):
                if current not in self.graph or self.graph.out_degree(current) == 0:
                    break
                successors = sorted(
                    self.graph.successors(current),
                    key=lambda nxt: (-self.graph[current][nxt].get("weight", 1), str(nxt)),
                )
                nxt = successors[0]
                edge = self.graph[current][nxt]
                path.append(f"{current} -[{edge.get('relation', 'next')}]-> {nxt}")
                current = str(nxt)

        contradictions = self.consistency_check(facts)
        answer = " ".join(f"[{idx + 1}] {fact.sentence}" for idx, fact in enumerate(facts)) if facts else "No verified facts found."
        return {
            "facts": facts,
            "reasoning_path": path or ["No strong graph path"],
            "contradictions": contradictions,
            "answer": answer,
            "verified": bool(facts) and not contradictions,
        }

    def consistency_check(self, facts: Sequence[VerifiedFact]) -> List[str]:
        contradictions = []
        grouped: Dict[str, set] = defaultdict(set)
        for fact in facts:
            if fact.relation in {"is", "are"}:
                grouped[fact.subject].add(fact.obj)

        for subject, objects in grouped.items():
            if len(objects) > 1:
                contradictions.append(f"conflicting definitions for {subject}: {sorted(objects)}")
        return contradictions


class GraphSpectralMemory:
    """Chunk retrieval with torch embeddings plus NetworkX Laplacian smoothing."""

    def __init__(self, encoder: TorchEmbeddingEncoder, chunk_size: int = 100):
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.graph = nx.Graph()
        self.spectral_embedding: Optional[np.ndarray] = None

    def add_text(self, text: str) -> None:
        text = _normalize_text(text)
        if not text:
            return
        stride = max(1, self.chunk_size // 2)
        windows = [text[i : i + self.chunk_size] for i in range(0, len(text), stride)]
        self.chunks.extend([chunk for chunk in windows if len(chunk) > 20])
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.chunks:
            self.embeddings = None
            self.graph.clear()
            self.spectral_embedding = None
            return

        self.embeddings = self.encoder.encode(self.chunks)
        self.graph = nx.Graph()
        for idx, chunk in enumerate(self.chunks):
            self.graph.add_node(idx, text=chunk)

        sims = self.embeddings @ self.embeddings.T
        for idx in range(len(self.chunks)):
            neighbor_idx = np.argsort(sims[idx])[-4:-1]
            for nxt in neighbor_idx:
                if idx == nxt:
                    continue
                weight = float(max(sims[idx, nxt], 0.0))
                if weight > 0:
                    self.graph.add_edge(idx, int(nxt), weight=weight)

        if self.graph.number_of_nodes() <= 1:
            self.spectral_embedding = np.zeros((len(self.chunks), 1), dtype=float)
            return

        laplacian = nx.normalized_laplacian_matrix(self.graph, weight="weight").astype(float).toarray()
        evals, evecs = np.linalg.eigh(laplacian)
        components = min(4, max(1, len(self.chunks) - 1))
        self.spectral_embedding = evecs[:, 1 : 1 + components] if len(self.chunks) > 1 else evecs[:, :1]

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, object]]:
        if self.embeddings is None or self.spectral_embedding is None:
            return []

        qvec = self.encoder.encode([query])[0]
        direct = self.embeddings @ qvec
        spectral_anchor = self.spectral_embedding[np.argmax(direct)]
        spectral_score = self.spectral_embedding @ spectral_anchor
        combined = 0.75 * direct + 0.25 * spectral_score
        top_idx = np.argsort(combined)[-top_k:][::-1]
        return [
            {
                "chunk": self.chunks[int(idx)],
                "score": float(combined[int(idx)]),
            }
            for idx in top_idx
        ]


class ConstrainedTokenMarkov:
    """Simple token Markov generator constrained by retrieved symbolic evidence."""

    def __init__(self):
        self.transitions: Dict[str, Counter] = defaultdict(Counter)
        self.starts: List[str] = []
        self.trained = False

    def fit(self, text: str) -> "ConstrainedTokenMarkov":
        tokens = _tokenize(text)
        if len(tokens) < 2:
            raise ValueError("Text corpus is too small for Markov training.")
        for left, right in zip(tokens, tokens[1:]):
            self.transitions[left][right] += 1
        self.starts = tokens[: max(1, min(50, len(tokens)))]
        self.trained = True
        return self

    def generate(self, prompt: str, verified_facts: Sequence[VerifiedFact], max_tokens: int = 48) -> str:
        if not self.trained:
            return "Model not trained."

        prompt_tokens = _tokenize(prompt)
        current = prompt_tokens[-1] if prompt_tokens else (self.starts[0] if self.starts else "model")
        allowed = set(prompt_tokens)
        for fact in verified_facts:
            allowed.update(_tokenize(fact.content))
            allowed.update(_tokenize(fact.sentence))

        generated = prompt_tokens[:] if prompt_tokens else [current]
        for _ in range(max_tokens):
            choices = self.transitions.get(current)
            if not choices:
                break
            ranked = sorted(choices.items(), key=lambda item: (-item[1], item[0]))
            next_token = None
            for token, _count in ranked:
                if not allowed or token in allowed:
                    next_token = token
                    break
            if next_token is None:
                next_token = ranked[0][0]
            generated.append(next_token)
            current = next_token
            if next_token.endswith((".", "!", "?")):
                break
        return " ".join(generated).strip()


class TinyTorchClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InterpretableDecisionEnsemble:
    """Torch + manual tree + KNN routing ensemble for low-resource routing."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.labels = np.array(["reasoning", "creative", "analysis"])
        self.model = TinyTorchClassifier(input_dim=7, output_dim=len(self.labels))
        self.knn = KNearestNeighbors(k=3)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.trained = False

    def _features(self, query: str) -> np.ndarray:
        lower = query.lower()
        tokens = _tokenize(query)
        return np.array(
            [
                len(tokens),
                sum(term in lower for term in ["why", "how", "explain", "logic", "prove"]),
                sum(term in lower for term in ["write", "story", "poem", "creative", "style"]),
                sum(term in lower for term in ["code", "bug", "function", "debug", "test"]),
                sum(term in lower for term in ["graph", "spectral", "memory", "markov"]),
                float(query.count("?")),
                float(any(ch.isdigit() for ch in query)),
            ],
            dtype=np.float32,
        )

    def fit(self) -> "InterpretableDecisionEnsemble":
        samples = [
            ("why do symbolic graphs improve logic", "reasoning"),
            ("explain the correctness of this code", "reasoning"),
            ("write a short creative paragraph", "creative"),
            ("compose a stylized note about graphs", "creative"),
            ("summarize the memory pipeline", "analysis"),
            ("analyze spectral retrieval performance", "analysis"),
        ]
        label_map = {label: idx for idx, label in enumerate(self.labels)}
        self.X_train = np.vstack([self._features(text) for text, _ in samples])
        self.y_train = np.array([label_map[label] for _, label in samples], dtype=np.int64)
        self.knn.fit(self.X_train, self.y_train)

        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.long)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.03)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(120):
            optimizer.zero_grad()
            logits = self.model(X_tensor)
            loss = loss_fn(logits, y_tensor)
            loss.backward()
            optimizer.step()
        self.model.eval()
        self.trained = True
        return self

    def _manual_tree(self, features: np.ndarray) -> Tuple[int, List[str]]:
        path = []
        if features[1] >= 1 or features[3] >= 1:
            path.append("reasoning_or_code_terms >= 1")
            return 0, path
        if features[2] >= 1:
            path.append("creative_terms >= 1")
            return 1, path
        path.append("default_analysis")
        return 2, path

    def predict(self, query: str) -> Dict[str, object]:
        if not self.trained:
            self.fit()

        features = self._features(query)
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.tensor(features, dtype=torch.float32).unsqueeze(0)), dim=1).cpu().numpy()[0]
        neural_idx = int(np.argmax(probs))
        tree_idx, tree_path = self._manual_tree(features)
        knn_idx = int(self.knn.predict(features.reshape(1, -1))[0])

        votes = [neural_idx, tree_idx, knn_idx]
        final_idx = Counter(votes).most_common(1)[0][0]
        return {
            "label": self.labels[final_idx],
            "votes": {
                "torch": self.labels[neural_idx],
                "tree": self.labels[tree_idx],
                "knn": self.labels[knn_idx],
            },
            "tree_path": tree_path,
            "torch_probs": {label: float(prob) for label, prob in zip(self.labels, probs)},
        }


class NeuralSymbolicHybridAgent:
    """Opt-in hybrid agent using torch embeddings, NetworkX reasoning, and constrained Markov generation."""

    def __init__(self, random_state: int = 42):
        self.encoder = TorchEmbeddingEncoder(random_state=random_state)
        self.memory = GraphSpectralMemory(self.encoder)
        self.reasoner = NetworkXSymbolicReasoner()
        self.markov = ConstrainedTokenMarkov()
        self.ensemble = InterpretableDecisionEnsemble(random_state=random_state).fit()
        self.learned = False

    def learn(self, corpus: str) -> None:
        self.memory.add_text(corpus)
        self.reasoner.fit(corpus)
        self.markov.fit(corpus)
        self.learned = True

    def think(self, query: str, max_tokens: int = 80) -> Dict[str, object]:
        if not self.learned:
            raise RuntimeError("Call learn(corpus) before think().")

        route = self.ensemble.predict(query)
        retrieved = self.memory.retrieve(query, top_k=4)
        reasoning = self.reasoner.infer(query, top_k=4)
        generated = self.markov.generate(query, reasoning["facts"], max_tokens=max_tokens)

        if reasoning["verified"]:
            answer = reasoning["answer"]
        else:
            answer = generated

        return {
            "query": query,
            "route": route,
            "retrieved": retrieved,
            "reasoning": {
                "path": reasoning["reasoning_path"],
                "facts": [fact.sentence for fact in reasoning["facts"]],
                "contradictions": reasoning["contradictions"],
                "verified": reasoning["verified"],
            },
            "generated": generated,
            "answer": answer,
        }