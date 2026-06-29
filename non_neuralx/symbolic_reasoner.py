import re
from collections import defaultdict


class SymbolicReasoner:
    """Lightweight symbolic next-concept reasoner from text co-occurrence."""

    def __init__(self):
        self.graph = defaultdict(set)
        self.edge_counts = defaultdict(int)
        self.knowledge = []
        self.entity_index = defaultdict(list)

    def fit(self, text: str) -> "SymbolicReasoner":
        sentences = re.split(r"[.!?]+", text)
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue

            words = re.findall(r"\b\w+\b", clean_sentence.lower())
            for i in range(len(words) - 1):
                self.graph[words[i]].add(words[i + 1])
                self.edge_counts[(words[i], words[i + 1])] += 1

            for fact in self._extract_facts(clean_sentence):
                self.knowledge.append(fact)
                self.entity_index[fact["subject"]].append(fact)
                self.entity_index[fact["object"]].append(fact)
        return self

    def _extract_facts(self, sentence: str):
        patterns = [
            ("is", r"^(.+?)\s+is\s+(?:an?|the)?\s*(.+)$"),
            ("are", r"^(.+?)\s+are\s+(?:an?|the)?\s*(.+)$"),
            ("uses", r"^(.+?)\s+uses\s+(.+)$"),
            ("supports", r"^(.+?)\s+supports\s+(.+)$"),
            ("reveals", r"^(.+?)\s+reveals?\s+(.+)$"),
            ("prevents", r"^(.+?)\s+prevents\s+(.+)$"),
        ]

        facts = []
        lowered = sentence.strip()
        for relation, pattern in patterns:
            match = re.match(pattern, lowered, flags=re.IGNORECASE)
            if not match:
                continue
            subject = re.sub(r"\s+", " ", match.group(1).strip().lower())
            obj = re.sub(r"\s+", " ", match.group(2).strip().lower())
            if len(subject) < 2 or len(obj) < 2:
                continue
            facts.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "sentence": sentence.strip(),
                    "verified": True,
                }
            )
        return facts

    def retrieve_facts(self, query: str, top_k: int = 4):
        tokens = set(re.findall(r"\b\w+\b", query.lower()))
        if not tokens:
            return []

        scored = []
        for fact in self.knowledge:
            haystack = f"{fact['subject']} {fact['relation']} {fact['object']}"
            hay_tokens = set(re.findall(r"\b\w+\b", haystack))
            overlap = len(tokens & hay_tokens)
            if overlap == 0:
                continue
            scored.append((overlap, fact["sentence"], fact))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [fact for _, _, fact in scored[:top_k]]

    def reason_about(self, query: str, top_k: int = 4, max_steps: int = 5):
        words = re.findall(r"\b\w+\b", query.lower())
        if not words:
            return {
                "chain": ["No concepts detected"],
                "facts": [],
                "answer": "No supported answer available.",
                "verified": False,
            }

        facts = self.retrieve_facts(query, top_k=top_k)
        chain = []
        current = words[0]

        for _ in range(max_steps):
            neighbors = sorted(
                self.graph.get(current, []),
                key=lambda item: (-self.edge_counts[(current, item)], item),
            )
            if not neighbors:
                break
            nxt = neighbors[0]
            chain.append(f"{current} -> {nxt}")
            current = nxt

        if facts:
            answer = " ".join(f"[{idx + 1}] {fact['sentence']}" for idx, fact in enumerate(facts))
        elif chain:
            answer = "Reasoning chain: " + " | ".join(chain)
        else:
            answer = "No supported answer available."

        return {
            "chain": chain if chain else ["No strong inference path"],
            "facts": facts,
            "answer": answer,
            "verified": bool(facts),
        }

    def reason(self, query: str, max_steps: int = 5):
        return self.reason_about(query, max_steps=max_steps)["chain"]
