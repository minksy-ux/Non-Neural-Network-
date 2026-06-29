import pickle
import random
import re
from collections import Counter, defaultdict

import numpy as np


class CharacterMarkovGenerator:
    """Character-level non-neural language model with optional backoff."""

    def __init__(self, order: int = 10, temperature: float = 0.95, backoff: bool = True, random_state: int = 42):
        self.order = order
        self.temperature = temperature
        self.backoff = backoff
        self.random_state = random_state

        # transitions[o][context] -> Counter(next_char)
        self.transitions = defaultdict(lambda: defaultdict(Counter))
        self.vocab = set()
        self.start_contexts = []
        self.trained = False

    def fit(self, text: str, verbose: bool = True) -> "CharacterMarkovGenerator":
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        text = re.sub(r"\s+", " ", text.strip())
        if len(text) <= 1:
            raise ValueError("Text corpus is too small for training.")

        self.vocab = set(text)
        n = len(text)

        for o in range(1, self.order + 1):
            for i in range(n - o):
                context = text[i : i + o]
                next_char = text[i + o]
                self.transitions[o][context][next_char] += 1
                if o == self.order and (i == 0 or text[i - 1] in ".!?"):
                    self.start_contexts.append(context)

        self.trained = True
        if verbose:
            print(f"CharacterMarkovGenerator trained | order={self.order} | chars={n}")
        return self

    def _sample_from_counter(self, counter: Counter) -> str:
        chars = list(counter.keys())
        counts = np.array(list(counter.values()), dtype=float)

        if self.temperature != 1.0:
            logits = counts / max(self.temperature, 1e-8)
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
        else:
            probs = counts / counts.sum()

        return str(np.random.choice(chars, p=probs))

    def _get_next_char(self, context: str) -> str:
        if not self.trained:
            return " "

        start_order = min(len(context), self.order)
        if not self.backoff:
            start_order = self.order

        for o in range(start_order, 0, -1):
            ctx = context[-o:]
            if ctx in self.transitions[o]:
                return self._sample_from_counter(self.transitions[o][ctx])

        return random.choice(list(self.vocab)) if self.vocab else " "

    def generate(self, prompt: str = "", max_length: int = 600, stop_at_punctuation: bool = True) -> str:
        if not self.trained:
            return "Model not trained."

        if prompt:
            context = prompt[-self.order :].ljust(self.order)
        else:
            fallback = list(self.transitions[self.order].keys())
            context = random.choice(self.start_contexts or fallback)

        generated = list(context)
        for _ in range(max_length):
            next_char = self._get_next_char("".join(generated))
            generated.append(next_char)
            if stop_at_punctuation and next_char in ".!?" and len(generated) > 40:
                break

        return re.sub(r"\s+", " ", "".join(generated)).strip()

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "transitions": dict(self.transitions),
                    "order": self.order,
                    "temperature": self.temperature,
                    "backoff": self.backoff,
                    "vocab": self.vocab,
                    "start_contexts": self.start_contexts,
                },
                f,
            )

    def load(self, filepath: str) -> "CharacterMarkovGenerator":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.transitions = defaultdict(lambda: defaultdict(Counter), data["transitions"])
        self.order = data["order"]
        self.temperature = data.get("temperature", self.temperature)
        self.backoff = data.get("backoff", self.backoff)
        self.vocab = data["vocab"]
        self.start_contexts = data["start_contexts"]
        self.trained = True
        return self
