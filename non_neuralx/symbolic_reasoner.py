import re
from collections import defaultdict


class SymbolicReasoner:
    """Lightweight symbolic next-concept reasoner from text co-occurrence."""

    def __init__(self):
        self.graph = defaultdict(set)

    def fit(self, text: str) -> "SymbolicReasoner":
        sentences = re.split(r"[.!?]+", text)
        for sentence in sentences:
            words = re.findall(r"\b\w+\b", sentence.lower())
            for i in range(len(words) - 1):
                self.graph[words[i]].add(words[i + 1])
        return self

    def reason(self, query: str, max_steps: int = 5):
        words = re.findall(r"\b\w+\b", query.lower())
        if not words:
            return ["No concepts detected"]

        trace = []
        current = words[0]
        for _ in range(max_steps):
            if current in self.graph and self.graph[current]:
                nxt = sorted(self.graph[current])[0]
                trace.append(f"{current} -> {nxt}")
                current = nxt
            else:
                break

        return trace if trace else ["No strong inference path"]
