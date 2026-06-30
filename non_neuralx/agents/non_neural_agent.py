import copy
import pickle
from collections import Counter, defaultdict
from functools import lru_cache
import time
from typing import Any, Dict, List

from ..config import AgentConfig
from .base_agent import BaseAgent
from .components import (
    BaseGenerator,
    BaseMemory,
    BaseReasoner,
    BaseRouter,
    CharacterMarkovComponent,
    KeywordRouter,
    SpectralMemoryComponent,
    SymbolicReasonerComponent,
)


class NonNeuralAgentV2(BaseAgent):
    """Refactored non-neural agent with explicit component interfaces and trace."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        memory: BaseMemory | None = None,
        reasoner: BaseReasoner | None = None,
        generator: BaseGenerator | None = None,
        router: BaseRouter | None = None,
    ):
        self.config = config or AgentConfig()
        self.memory = memory or SpectralMemoryComponent()
        self.reasoner = reasoner or SymbolicReasonerComponent()
        self.generator = generator or CharacterMarkovComponent()
        self.router = router or KeywordRouter()
        self.trace: List[Dict[str, Any]] = []
        self._setup_caches()

    def _setup_caches(self) -> None:
        @lru_cache(maxsize=256)
        def _retrieve_cached(query: str) -> tuple[str, ...]:
            result = self.memory.retrieve(query, top_k=self.config.retrieval_top_k)
            return tuple(result)

        @lru_cache(maxsize=256)
        def _reason_cached(query: str) -> Dict[str, Any]:
            result = self.reasoner.reason_about(
                query,
                top_k=self.config.reasoning_top_k,
                max_steps=self.config.reasoning_max_steps,
            )
            return copy.deepcopy(result)

        self._retrieve_cached = _retrieve_cached
        self._reason_cached = _reason_cached

    def _clear_caches(self) -> None:
        self._retrieve_cached.cache_clear()
        self._reason_cached.cache_clear()

    def learn(self, corpus: str) -> None:
        self._clear_caches()
        self.memory.add_text(corpus)
        self.reasoner.fit(corpus)
        self.generator.fit(corpus)

    def think(self, query: str) -> Dict[str, Any]:
        self.trace.clear()
        t_start = time.perf_counter()
        route, route_confidence = self.router.route_with_confidence(query)

        t0 = time.perf_counter()
        retrieved = list(self._retrieve_cached(query))
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        self.trace.append({"step": "retrieval", "content": retrieved, "latency_ms": retrieval_ms})

        t1 = time.perf_counter()
        reasoned = self._reason_cached(query)
        reasoning_ms = (time.perf_counter() - t1) * 1000.0
        self.trace.append({"step": "reasoning", "content": reasoned, "latency_ms": reasoning_ms})

        verified = bool(reasoned.get("verified", False))
        t2 = time.perf_counter()
        if (route in {"reasoning", "code"} and verified) or (route_confidence < 0.55 and verified):
            answer = str(reasoned.get("answer", "No supported answer available."))
            generation_mode = "reasoned"
        else:
            answer = self.generator.generate(query, max_length=self.config.max_generation_length)
            generation_mode = "generated"
        generation_ms = (time.perf_counter() - t2) * 1000.0

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self.trace.append(
            {
                "step": "response",
                "mode": generation_mode,
                "latency_ms": generation_ms,
                "total_latency_ms": total_ms,
            }
        )

        confidence = self._compute_confidence(reasoned, retrieved, route_confidence)
        return {
            "query": query,
            "route": route,
            "route_confidence": route_confidence,
            "answer": answer,
            "reasoning_trace": self.trace.copy(),
            "confidence": confidence,
            "timings_ms": {
                "retrieval": retrieval_ms,
                "reasoning": reasoning_ms,
                "response": generation_ms,
                "total": total_ms,
            },
        }

    def _compute_confidence(self, reasoned: Dict[str, Any], retrieved: List[str], route_confidence: float) -> float:
        score = 0.2
        if bool(reasoned.get("verified", False)):
            score += 0.5
        if retrieved:
            score += 0.2
        if reasoned.get("chain"):
            score += 0.1
        score = 0.8 * score + 0.2 * max(0.0, min(1.0, route_confidence))
        return min(1.0, score)

    def save(self, filepath: str) -> None:
        memory_impl = self.memory.impl
        reasoner_impl = self.reasoner.impl
        generator_impl = self.generator.impl
        serialized_transitions = {
            order: {context: dict(counter) for context, counter in by_context.items()}
            for order, by_context in generator_impl.transitions.items()
        }
        payload = {
            "config": self.config,
            "trace": self.trace,
            "memory": {
                "chunk_size": memory_impl.chunk_size,
                "vector_size": memory_impl.vector_size,
                "chunks": list(memory_impl.chunks),
                "embeddings": memory_impl.embeddings,
            },
            "reasoner": {
                "graph": dict(reasoner_impl.graph),
                "edge_counts": dict(reasoner_impl.edge_counts),
                "knowledge": list(reasoner_impl.knowledge),
                "entity_index": dict(reasoner_impl.entity_index),
            },
            "generator": {
                "order": generator_impl.order,
                "temperature": generator_impl.temperature,
                "backoff": generator_impl.backoff,
                "random_state": generator_impl.random_state,
                "transitions": serialized_transitions,
                "vocab": list(generator_impl.vocab),
                "start_contexts": list(generator_impl.start_contexts),
                "trained": generator_impl.trained,
            },
        }
        with open(filepath, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, filepath: str) -> "NonNeuralAgentV2":
        with open(filepath, "rb") as handle:
            payload = pickle.load(handle)

        agent = cls(config=payload["config"])
        agent.trace = list(payload.get("trace", []))

        memory_state = payload["memory"]
        agent.memory.impl.chunk_size = memory_state["chunk_size"]
        agent.memory.impl.vector_size = memory_state["vector_size"]
        agent.memory.impl.chunks = list(memory_state["chunks"])
        agent.memory.impl.embeddings = memory_state["embeddings"]

        reasoner_state = payload["reasoner"]
        agent.reasoner.impl.graph = defaultdict(set, reasoner_state["graph"])
        agent.reasoner.impl.edge_counts = defaultdict(int, reasoner_state["edge_counts"])
        agent.reasoner.impl.knowledge = list(reasoner_state["knowledge"])
        agent.reasoner.impl.entity_index = defaultdict(list, reasoner_state["entity_index"])

        generator_state = payload["generator"]
        transitions = defaultdict(lambda: defaultdict(Counter))
        for order, by_context in generator_state["transitions"].items():
            transitions[int(order)] = defaultdict(Counter)
            for context, counts in by_context.items():
                transitions[int(order)][context] = Counter(counts)
        agent.generator.impl.transitions = transitions
        agent.generator.impl.order = int(generator_state["order"])
        agent.generator.impl.temperature = float(generator_state["temperature"])
        agent.generator.impl.backoff = bool(generator_state["backoff"])
        agent.generator.impl.random_state = int(generator_state["random_state"])
        agent.generator.impl.vocab = set(generator_state["vocab"])
        agent.generator.impl.start_contexts = list(generator_state["start_contexts"])
        agent.generator.impl.trained = bool(generator_state["trained"])

        agent._setup_caches()
        return agent
